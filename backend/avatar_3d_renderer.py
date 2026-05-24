"""
3D Avatar Renderer for lean_server.py
=====================================
Standalone (no FastAPI deps) helper that exposes:
  - get_or_render_avatar_3d(token) -> Path | None
  - stitch_avatar_videos(tokens, out_path) -> Path | None

Uses the DigiHuman-style retargeting pipeline. Subprocess-based to keep
the lean server lightweight (no heavyweight imports at startup).
"""
from __future__ import annotations

import subprocess
import json
from pathlib import Path
from typing import Optional

PLATFORM_ROOT = Path(__file__).parent.parent  # esl-platform/
SCRIPTS_DIR = PLATFORM_ROOT / "scripts" / "animate"
DATA_DIR = PLATFORM_ROOT / "data"
AVATAR_GLB = DATA_DIR / "avatars" / "arab-man" / "source" / "ready player me arab sheik.glb"
MOTION_DB_DIR = DATA_DIR / "motion_db"
HOLISTIC_CACHE = DATA_DIR / "processed" / "mocap_holistic_v2"
ANIM_GLB_DIR = DATA_DIR / "avatars" / "arab-man"
# Alias for clarity from the server's perspective: this is where merged
# avatar+animation GLBs (arab_sheik_<TOKEN>.glb) are stored.
AVATAR_GLB_DIR = ANIM_GLB_DIR
RENDER_DIR = DATA_DIR / "avatar_videos_3d"
STITCHED_DIR = RENDER_DIR / "stitched"

# Render settings (matches build_digihuman_v2.sh)
FPS = 25
WIDTH = 600
HEIGHT = 700

HOLISTIC_CACHE.mkdir(parents=True, exist_ok=True)
RENDER_DIR.mkdir(parents=True, exist_ok=True)
STITCHED_DIR.mkdir(parents=True, exist_ok=True)


def _run(cmd: list[str], cwd: Optional[Path] = None) -> bool:
    """Run a subprocess and return True if it succeeded."""
    try:
        result = subprocess.run(
            cmd, cwd=str(cwd) if cwd else None,
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print(f"[Avatar3D] FAILED: {' '.join(cmd[:3])}... → {result.stderr[-300:]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"[Avatar3D] TIMEOUT: {' '.join(cmd[:3])}...")
        return False
    except Exception as e:
        print(f"[Avatar3D] ERROR: {e}")
        return False


def get_or_render_avatar_3d(token: str) -> Optional[Path]:
    """Render the 3D avatar video for a single gloss token, or return cached.
    Returns the path to the MP4 (rendered or cached), or None if the token
    isn't available in the motion database.
    """
    token = token.upper()
    out_mp4 = RENDER_DIR / f"arab_sheik_{token}.mp4"

    # Cached?
    if out_mp4.exists() and out_mp4.stat().st_size > 5000:
        return out_mp4

    source_video = MOTION_DB_DIR / f"{token}.mp4"
    if not source_video.exists():
        return None

    holistic_json = HOLISTIC_CACHE / f"{token}.json"
    anim_glb = ANIM_GLB_DIR / f"_{token}_anim.glb"
    merged_glb = ANIM_GLB_DIR / f"arab_sheik_{token}.glb"

    # Step 1: extract holistic landmarks (cached)
    if not holistic_json.exists():
        if not _run([
            "python3", str(SCRIPTS_DIR / "extract_v2.py"),
            str(source_video), str(holistic_json),
        ]):
            return None

    # Step 2: retarget
    if not anim_glb.exists() or anim_glb.stat().st_mtime < holistic_json.stat().st_mtime:
        if not _run([
            "python3", str(SCRIPTS_DIR / "retarget_digihuman.py"),
            str(AVATAR_GLB), str(holistic_json), str(anim_glb),
            "--smooth", "5", "--trim-trailing",
        ]):
            return None

    # Step 3: merge animation into avatar GLB
    if not merged_glb.exists() or merged_glb.stat().st_mtime < anim_glb.stat().st_mtime:
        if not _run([
            "python3", str(SCRIPTS_DIR / "merge_animation.py"),
            str(AVATAR_GLB), str(anim_glb), str(merged_glb),
        ]):
            return None

    # Step 4: render to MP4
    if not _run([
        "node", str(SCRIPTS_DIR / "render.js"),
        str(merged_glb), str(out_mp4),
        "--fps", str(FPS), "--w", str(WIDTH), "--h", str(HEIGHT),
    ]):
        return None

    return out_mp4 if out_mp4.exists() else None


def stitch_avatar_videos(tokens: list[str], out_path: Path) -> Optional[Path]:
    """Render each token then concatenate via ffmpeg. Returns out_path or None."""
    clips: list[Path] = []
    for t in tokens[:10]:  # cap at 10 tokens
        clip = get_or_render_avatar_3d(t)
        if clip:
            clips.append(clip)
        else:
            print(f"[Avatar3D] skipping missing token: {t}")

    if not clips:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(clips) == 1:
        import shutil
        shutil.copyfile(clips[0], out_path)
        return out_path

    # ffmpeg concat demuxer
    list_file = out_path.parent / f"{out_path.stem}.txt"
    with open(list_file, "w") as f:
        for c in clips:
            f.write(f"file '{c.resolve()}'\n")
    try:
        if not _run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(out_path),
        ]):
            return None
    finally:
        list_file.unlink(missing_ok=True)

    return out_path if out_path.exists() else None
