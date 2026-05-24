"""
3D Avatar Renderer (DigiHuman-style)
====================================
Renders the Arab sheikh GLTF avatar with sign-language motion retargeted from
MediaPipe Holistic 3D landmarks.

Pipeline:
  1. For each gloss token, look up the source video in data/motion_db/{TOKEN}.mp4
  2. Extract MediaPipe landmarks (new Tasks API: PoseLandmarker + HandLandmarker)
     -> data/processed/mocap_holistic_v2/{TOKEN}.json
  3. Retarget to the Arab sheikh GLB using the DigiHuman LookRotation algorithm
     -> Per-bone glTF animation track GLB
  4. Merge the animation into the avatar GLB
  5. Render to MP4 via headless Chromium + Three.js

For multi-token sequences, render each token separately then concatenate via ffmpeg.

The actual implementation calls the scripts in scripts/animate/ as subprocesses
since they're already battle-tested.
"""
from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

PLATFORM_ROOT = Path(__file__).parent.parent.parent.parent  # esl-platform/
SCRIPTS_DIR = PLATFORM_ROOT / "scripts" / "animate"
DATA_DIR = PLATFORM_ROOT / "data"
AVATAR_GLB = DATA_DIR / "avatars" / "arab-man" / "source" / "ready player me arab sheik.glb"


@dataclass
class Render3DConfig:
    width: int = 600
    height: int = 700
    fps: int = 25
    smooth: int = 5
    trim_trailing: bool = True
    output_dir: Optional[Path] = None


@dataclass
class Render3DResult:
    output_path: Path
    duration_seconds: float
    file_size_bytes: int
    tokens_rendered: list[str]
    tokens_missing: list[str]


class Avatar3DRenderer:
    """Renders ESL motion to MP4 using the 3D GLB avatar pipeline."""

    def __init__(self, config: Optional[Render3DConfig] = None) -> None:
        self.config = config or Render3DConfig()
        self.settings = get_settings()
        if self.config.output_dir is None:
            self.config.output_dir = self.settings.VIDEO_OUTPUT_DIR
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    async def render_token(self, token: str) -> Optional[Path]:
        """Render a single gloss token to an MP4. Returns path or None if not available."""
        source_video = DATA_DIR / "motion_db" / f"{token}.mp4"
        if not source_video.exists():
            logger.warning("token_video_missing", token=token, path=str(source_video))
            return None

        holistic_json = DATA_DIR / "processed" / "mocap_holistic_v2" / f"{token}.json"
        anim_glb = DATA_DIR / "avatars" / "arab-man" / f"_{token}_anim.glb"
        merged_glb = DATA_DIR / "avatars" / "arab-man" / f"arab_sheik_{token}.glb"
        out_mp4 = self.config.output_dir / f"arab_sheik_{token}.mp4"

        holistic_json.parent.mkdir(parents=True, exist_ok=True)

        # Step 1: extract holistic if not cached
        if not holistic_json.exists():
            await self._run([
                "python3", str(SCRIPTS_DIR / "extract_v2.py"),
                str(source_video), str(holistic_json),
            ])

        # Step 2: retarget
        retarget_args = [
            "python3", str(SCRIPTS_DIR / "retarget_digihuman.py"),
            str(AVATAR_GLB), str(holistic_json), str(anim_glb),
            "--smooth", str(self.config.smooth),
        ]
        if self.config.trim_trailing:
            retarget_args.append("--trim-trailing")
        await self._run(retarget_args)

        # Step 3: merge anim into avatar GLB
        await self._run([
            "python3", str(SCRIPTS_DIR / "merge_animation.py"),
            str(AVATAR_GLB), str(anim_glb), str(merged_glb),
        ])

        # Step 4: render to MP4
        await self._run([
            "node", str(SCRIPTS_DIR / "render.js"),
            str(merged_glb), str(out_mp4),
            "--fps", str(self.config.fps),
            "--w", str(self.config.width),
            "--h", str(self.config.height),
        ])

        return out_mp4 if out_mp4.exists() else None

    async def render_sequence(self, tokens: list[str], output_name: str | None = None) -> Render3DResult:
        """Render a sequence of gloss tokens, concatenating into a single MP4."""
        clips: list[Path] = []
        rendered_tokens: list[str] = []
        missing_tokens: list[str] = []

        for token in tokens:
            clip = await self.render_token(token)
            if clip:
                clips.append(clip)
                rendered_tokens.append(token)
            else:
                missing_tokens.append(token)

        if not clips:
            raise RuntimeError(f"No tokens could be rendered (missing: {missing_tokens})")

        # Concatenate via ffmpeg
        name = output_name or f"sequence_{'_'.join(rendered_tokens)}"
        out_path = self.config.output_dir / f"{name}.mp4"
        if len(clips) == 1:
            # Single clip — just copy/symlink
            import shutil
            shutil.copyfile(clips[0], out_path)
        else:
            await self._concat_videos(clips, out_path)

        size = out_path.stat().st_size
        duration = await self._probe_duration(out_path)
        return Render3DResult(
            output_path=out_path,
            duration_seconds=duration,
            file_size_bytes=size,
            tokens_rendered=rendered_tokens,
            tokens_missing=missing_tokens,
        )

    async def _concat_videos(self, clips: list[Path], out_path: Path) -> None:
        # Use ffmpeg concat demuxer
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            for c in clips:
                f.write(f"file '{c.resolve()}'\n")
            list_file = Path(f.name)
        try:
            await self._run([
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(list_file),
                "-c", "copy", str(out_path),
            ])
        finally:
            list_file.unlink(missing_ok=True)

    async def _probe_duration(self, path: Path) -> float:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        try:
            return float(stdout.decode().strip())
        except Exception:
            return 0.0

    async def _run(self, cmd: list[str]) -> None:
        logger.info("subprocess_run", cmd=" ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            err = stderr.decode(errors="replace")
            logger.error("subprocess_failed", cmd=cmd[0], code=proc.returncode, err=err[-500:])
            raise RuntimeError(f"{cmd[0]} failed (code {proc.returncode}): {err[-200:]}")
        logger.info("subprocess_ok", cmd=cmd[0])
