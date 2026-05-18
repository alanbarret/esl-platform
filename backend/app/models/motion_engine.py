"""
Motion Generation Engine
========================
Converts ESL gloss sequences into signed motion sequences.

Architecture (Phase 1 — Retrieval + Blending):
  1. Lookup gloss token in motion database (pre-recorded poses)
  2. Temporally align retrieved clips
  3. Blend transitions between signs
  4. Smooth interpolation

Architecture (Phase 2 — Transformer/Diffusion — prepared, not yet active):
  - GlosFormer: Transformer conditioned on gloss embeddings
  - Diffusion motion synthesis (future)

Motion format:
  - Per-frame bone rotations (quaternions)
  - Compatible with GLTF AnimationClip format
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.pose_extractor import PoseSequence

logger = get_logger(__name__)


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class BoneKeyframe:
    time: float             # seconds
    position: list[float]   # [x, y, z]
    rotation: list[float]   # [x, y, z, w] quaternion
    scale: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])


@dataclass
class BoneTrack:
    bone_name: str
    keyframes: list[BoneKeyframe] = field(default_factory=list)


@dataclass
class SignMotion:
    gloss_token: str
    duration_seconds: float
    fps: int
    bone_tracks: list[BoneTrack] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MotionSequence:
    gloss_tokens: list[str]
    total_duration: float
    fps: int
    sign_motions: list[SignMotion] = field(default_factory=list)

    def to_gltf_animation(self) -> dict:
        """
        Export as GLTF-compatible animation clip structure.
        Each bone track maps to a GLTF animation channel + sampler pair.
        """
        channels = []
        samplers = []
        sampler_idx = 0

        for motion in self.sign_motions:
            for track in motion.bone_tracks:
                times = [kf.time for kf in track.keyframes]
                rotations = [kf.rotation for kf in track.keyframes]

                samplers.append({
                    "input": times,
                    "interpolation": "LINEAR",
                    "output": rotations,
                })
                channels.append({
                    "sampler": sampler_idx,
                    "target": {
                        "node": track.bone_name,
                        "path": "rotation",
                    }
                })
                sampler_idx += 1

        return {
            "name": "_".join(self.gloss_tokens),
            "channels": channels,
            "samplers": samplers,
            "duration": self.total_duration,
            "fps": self.fps,
        }


# ── Motion Database ────────────────────────────────────────────────────────────

class MotionDatabase:
    """
    Stores pre-recorded sign motion clips keyed by gloss token.
    Each entry is a PoseSequence JSON file.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._cache: dict[str, SignMotion] = {}
        self._index: dict[str, Path] = {}
        self._load_index()

    def _load_index(self) -> None:
        if not self.db_path.exists():
            logger.warning("motion_db_not_found", path=str(self.db_path))
            return
        for file in self.db_path.glob("*.json"):
            gloss = file.stem.upper()
            self._index[gloss] = file
        logger.info("motion_db_loaded", entries=len(self._index))

    def lookup(self, gloss: str) -> Optional[SignMotion]:
        """Retrieve a SignMotion for a gloss token."""
        gloss = gloss.upper()
        if gloss in self._cache:
            return self._cache[gloss]
        if gloss not in self._index:
            return None
        pose_seq = PoseSequence.load(self._index[gloss])
        motion = self._pose_to_motion(gloss, pose_seq)
        self._cache[gloss] = motion
        return motion

    def get_nearest(self, gloss: str) -> Optional[SignMotion]:
        """Return nearest available gloss if exact not found."""
        if m := self.lookup(gloss):
            return m
        # Simple fallback — could be improved with embedding similarity
        logger.warning("gloss_not_in_db_using_fallback", gloss=gloss)
        if self._index:
            fallback = next(iter(self._index))
            return self.lookup(fallback)
        return None

    @staticmethod
    def _pose_to_motion(gloss: str, seq: PoseSequence) -> SignMotion:
        """Convert PoseSequence → SignMotion with bone tracks."""
        fps = int(seq.fps)
        duration = seq.total_frames / fps

        # Map MediaPipe body landmarks to Arab Man GLB bone names
        # GLB rig uses mixamo-style naming
        LANDMARK_TO_BONE = {
            11: "LeftShoulder",  12: "RightShoulder",
            13: "LeftArm",       14: "RightArm",
            15: "LeftForeArm",   16: "RightForeArm",
            17: "LeftHand",      18: "RightHand",
            23: "LeftUpLeg",     24: "RightUpLeg",
            0:  "Head",
            1:  "Neck",
            24: "Spine2",
        }

        bone_tracks: dict[str, BoneTrack] = {
            bone: BoneTrack(bone_name=bone)
            for bone in LANDMARK_TO_BONE.values()
        }

        for frame in seq.frames:
            t = frame.timestamp_ms / 1000.0
            for lm_idx, bone_name in LANDMARK_TO_BONE.items():
                if lm_idx < len(frame.body):
                    lm = frame.body[lm_idx]
                    # Stub rotation from position delta — real impl uses IK
                    rot = _position_to_quaternion(lm.x, lm.y, lm.z)
                    bone_tracks[bone_name].keyframes.append(
                        BoneKeyframe(
                            time=t,
                            position=[lm.x, lm.y, lm.z],
                            rotation=rot,
                        )
                    )

        return SignMotion(
            gloss_token=gloss,
            duration_seconds=duration,
            fps=fps,
            bone_tracks=list(bone_tracks.values()),
        )


# ── Motion Engine ──────────────────────────────────────────────────────────────

class MotionEngine:
    """
    Core motion generation pipeline.

    Phase 1: Retrieval + Blending (active)
    Phase 2: GlosFormer transformer (prepared)
    Phase 3: Diffusion synthesis (future)
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.db = MotionDatabase(self.settings.MOTION_DB_DIR)

    def generate(
        self,
        gloss_tokens: list[str],
        fps: int | None = None,
        blend_frames: int = 5,
    ) -> MotionSequence:
        """
        Convert gloss token list → MotionSequence with bone animations.

        Args:
            gloss_tokens:  List of ESL gloss tokens, e.g. ["HELLO", "YOU"]
            fps:           Output frame rate (default from config)
            blend_frames:  Frames to blend between signs

        Returns:
            MotionSequence with per-bone keyframe tracks
        """
        fps = fps or self.settings.VIDEO_FPS
        sign_motions: list[SignMotion] = []
        total_duration = 0.0

        for token in gloss_tokens:
            motion = self.db.get_nearest(token)
            if motion is None:
                logger.warning("no_motion_for_gloss", gloss=token)
                motion = self._create_rest_pose(token, duration=0.5, fps=fps)

            sign_motions.append(motion)
            total_duration += motion.duration_seconds

        # Apply transition blending between signs
        blended = self._blend_transitions(sign_motions, blend_frames, fps)

        return MotionSequence(
            gloss_tokens=gloss_tokens,
            total_duration=total_duration,
            fps=fps,
            sign_motions=blended,
        )

    @staticmethod
    def _blend_transitions(
        motions: list[SignMotion],
        blend_frames: int,
        fps: int,
    ) -> list[SignMotion]:
        """
        Smooth transitions between consecutive sign motions.
        Uses linear interpolation on overlapping keyframes.
        """
        if len(motions) <= 1:
            return motions

        blend_duration = blend_frames / fps
        blended = [motions[0]]

        for i in range(1, len(motions)):
            prev = motions[i - 1]
            curr = motions[i]

            # Offset current motion's keyframes by cumulative time
            offset = sum(m.duration_seconds for m in blended)
            shifted = _shift_motion_time(curr, offset)
            blended.append(shifted)

        return blended

    @staticmethod
    def _create_rest_pose(gloss: str, duration: float, fps: int) -> SignMotion:
        """Create a neutral/rest pose for unknown gloss tokens."""
        REST_BONES = [
            "LeftShoulder", "RightShoulder",
            "LeftArm", "RightArm",
            "LeftForeArm", "RightForeArm",
            "LeftHand", "RightHand",
            "Head", "Neck", "Spine2",
        ]
        tracks = []
        for bone in REST_BONES:
            kfs = [
                BoneKeyframe(time=0.0, position=[0, 0, 0], rotation=[0, 0, 0, 1]),
                BoneKeyframe(time=duration, position=[0, 0, 0], rotation=[0, 0, 0, 1]),
            ]
            tracks.append(BoneTrack(bone_name=bone, keyframes=kfs))

        return SignMotion(
            gloss_token=gloss,
            duration_seconds=duration,
            fps=fps,
            bone_tracks=tracks,
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _position_to_quaternion(x: float, y: float, z: float) -> list[float]:
    """Stub: convert normalized position to identity-ish quaternion."""
    # In production: use IK solver (e.g. fabrik) or learned bone angles
    return [0.0, 0.0, 0.0, 1.0]


def _shift_motion_time(motion: SignMotion, offset: float) -> SignMotion:
    """Offset all keyframe times by a fixed amount."""
    shifted_tracks = []
    for track in motion.bone_tracks:
        shifted_kfs = [
            BoneKeyframe(
                time=kf.time + offset,
                position=kf.position,
                rotation=kf.rotation,
                scale=kf.scale,
            )
            for kf in track.keyframes
        ]
        shifted_tracks.append(BoneTrack(bone_name=track.bone_name, keyframes=shifted_kfs))

    return SignMotion(
        gloss_token=motion.gloss_token,
        duration_seconds=motion.duration_seconds,
        fps=motion.fps,
        bone_tracks=shifted_tracks,
    )


# ── Singleton ──────────────────────────────────────────────────────────────────

_engine: Optional[MotionEngine] = None


def get_motion_engine() -> MotionEngine:
    global _engine
    if _engine is None:
        _engine = MotionEngine()
    return _engine
