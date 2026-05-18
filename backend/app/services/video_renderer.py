"""
Video Renderer
==============
Renders signed avatar animation to MP4 using OpenCV + FFmpeg.

Pipeline:
  1. Receive MotionSequence + avatar config
  2. Render skeleton overlay frames (Phase 1)
  3. Composite with background
  4. Encode to MP4 via FFmpeg

Phase 2: Off-screen Three.js rendering via headless browser (Puppeteer/Playwright)
Phase 3: Native GPU-accelerated rendering
"""
from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.motion_engine import MotionSequence, BoneKeyframe

logger = get_logger(__name__)


@dataclass
class RenderConfig:
    width: int = 1920
    height: int = 1080
    fps: int = 30
    background_color: tuple[int, int, int] = (18, 18, 18)
    transparent_bg: bool = False
    codec: str = "libx264"
    crf: int = 18         # Quality (lower = better, 0-51)
    preset: str = "fast"


@dataclass
class RenderResult:
    output_path: Path
    duration_seconds: float
    frame_count: int
    file_size_bytes: int


class VideoRenderer:
    """
    Renders MotionSequence to MP4.

    Phase 1 (active): Skeleton wireframe renderer using OpenCV.
    Phase 2 (prepared): Three.js headless renderer via Playwright.
    """

    # MediaPipe / pose skeleton connections (body only)
    SKELETON_CONNECTIONS = [
        (11, 12),  # shoulders
        (11, 13), (13, 15),  # left arm
        (12, 14), (14, 16),  # right arm
        (23, 24),  # hips
        (11, 23), (12, 24),  # torso
    ]

    # Named bone positions (normalized 0-1, scaled to frame)
    REST_POSITIONS = {
        "LeftShoulder":  (0.35, 0.25),
        "RightShoulder": (0.65, 0.25),
        "LeftUpperArm":  (0.28, 0.38),
        "RightUpperArm": (0.72, 0.38),
        "LeftLowerArm":  (0.22, 0.52),
        "RightLowerArm": (0.78, 0.52),
        "LeftHip":       (0.40, 0.62),
        "RightHip":      (0.60, 0.62),
    }

    def __init__(self, config: Optional[RenderConfig] = None) -> None:
        self.config = config or RenderConfig()
        self.settings = get_settings()

    async def render(
        self,
        motion: MotionSequence,
        output_path: Optional[Path] = None,
    ) -> RenderResult:
        """
        Render motion sequence to MP4.

        Args:
            motion:       MotionSequence from MotionEngine.generate()
            output_path:  Where to save MP4. Auto-generated if None.

        Returns:
            RenderResult with path, duration, frame count, file size.
        """
        if output_path is None:
            self.settings.VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            gloss_str = "_".join(motion.gloss_tokens[:5])
            output_path = self.settings.VIDEO_OUTPUT_DIR / f"{gloss_str}.mp4"

        logger.info("rendering_video",
                    gloss=motion.gloss_tokens,
                    duration=motion.total_duration,
                    output=str(output_path))

        # Phase 1: render skeleton wireframe
        frames = await asyncio.to_thread(self._render_frames, motion)
        await asyncio.to_thread(self._encode_video, frames, output_path)

        result = RenderResult(
            output_path=output_path,
            duration_seconds=motion.total_duration,
            frame_count=len(frames),
            file_size_bytes=output_path.stat().st_size if output_path.exists() else 0,
        )
        logger.info("render_complete", **result.__dict__)
        return result

    def _render_frames(self, motion: MotionSequence) -> list[np.ndarray]:
        """Render all frames as numpy arrays (BGR)."""
        cfg = self.config
        total_frames = int(motion.total_duration * cfg.fps)
        frames = []

        for frame_idx in range(total_frames):
            t = frame_idx / cfg.fps
            canvas = np.full((cfg.height, cfg.width, 3), cfg.background_color, dtype=np.uint8)
            self._draw_skeleton(canvas, motion, t)
            self._draw_overlay(canvas, motion, frame_idx, total_frames)
            frames.append(canvas)

        return frames

    def _draw_skeleton(
        self,
        canvas: np.ndarray,
        motion: MotionSequence,
        t: float,
    ) -> None:
        """Draw skeleton wireframe on canvas at time t."""
        h, w = canvas.shape[:2]

        # Get bone positions at time t (interpolated)
        positions: dict[str, tuple[int, int]] = {}
        for bone, rest_pos in self.REST_POSITIONS.items():
            # Find motion data for this bone
            rotation = self._get_rotation_at(motion, bone, t)
            # Apply simple FK offset based on rotation (stub)
            px = int(rest_pos[0] * w)
            py = int(rest_pos[1] * h)
            positions[bone] = (px, py)

        # Draw joints
        for name, pos in positions.items():
            cv2.circle(canvas, pos, 10, (168, 255, 75), -1)  # neon green

        # Draw limb connections
        connections = [
            ("LeftShoulder", "RightShoulder"),
            ("LeftShoulder", "LeftUpperArm"),
            ("LeftUpperArm", "LeftLowerArm"),
            ("RightShoulder", "RightUpperArm"),
            ("RightUpperArm", "RightLowerArm"),
            ("LeftHip", "RightHip"),
            ("LeftShoulder", "LeftHip"),
            ("RightShoulder", "RightHip"),
        ]
        for a, b in connections:
            if a in positions and b in positions:
                cv2.line(canvas, positions[a], positions[b], (124, 58, 237), 4)

        # Draw head
        head_x = int(0.5 * canvas.shape[1])
        head_y = int(0.12 * canvas.shape[0])
        cv2.circle(canvas, (head_x, head_y), 45, (200, 200, 200), 2)

    def _draw_overlay(
        self,
        canvas: np.ndarray,
        motion: MotionSequence,
        frame_idx: int,
        total_frames: int,
    ) -> None:
        """Draw gloss label and progress bar."""
        h, w = canvas.shape[:2]

        # Current sign label
        sign_idx = min(
            int(frame_idx / total_frames * len(motion.gloss_tokens)),
            len(motion.gloss_tokens) - 1,
        ) if motion.gloss_tokens else 0
        current_gloss = motion.gloss_tokens[sign_idx] if motion.gloss_tokens else ""

        cv2.putText(canvas, current_gloss, (w // 2 - 100, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (168, 255, 75), 4)

        # Progress bar
        progress = frame_idx / max(total_frames - 1, 1)
        bar_w = int(w * 0.8)
        bar_x = int(w * 0.1)
        bar_y = h - 20
        cv2.rectangle(canvas, (bar_x, bar_y - 6), (bar_x + bar_w, bar_y + 6),
                      (50, 50, 50), -1)
        cv2.rectangle(canvas, (bar_x, bar_y - 6),
                      (bar_x + int(bar_w * progress), bar_y + 6),
                      (124, 58, 237), -1)

    @staticmethod
    def _get_rotation_at(motion: MotionSequence, bone: str, t: float) -> list[float]:
        """Get interpolated rotation quaternion for a bone at time t."""
        for sign_motion in motion.sign_motions:
            for track in sign_motion.bone_tracks:
                if track.bone_name == bone and track.keyframes:
                    kfs = track.keyframes
                    # Find surrounding keyframes
                    for i in range(len(kfs) - 1):
                        if kfs[i].time <= t <= kfs[i + 1].time:
                            alpha = (t - kfs[i].time) / max(kfs[i + 1].time - kfs[i].time, 1e-6)
                            r0 = np.array(kfs[i].rotation)
                            r1 = np.array(kfs[i + 1].rotation)
                            return (r0 + alpha * (r1 - r0)).tolist()  # lerp (slerp in prod)
        return [0.0, 0.0, 0.0, 1.0]

    def _encode_video(self, frames: list[np.ndarray], output_path: Path) -> None:
        """Encode frames to MP4 using FFmpeg via temp rawvideo pipe."""
        cfg = self.config
        h, w = frames[0].shape[:2]

        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            for frame in frames:
                tmp.write(frame.tobytes())

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-pix_fmt", "bgr24",
            "-r", str(cfg.fps),
            "-i", str(tmp_path),
            "-c:v", cfg.codec,
            "-crf", str(cfg.crf),
            "-preset", cfg.preset,
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]

        logger.info("ffmpeg_encode", cmd=" ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        tmp_path.unlink(missing_ok=True)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
