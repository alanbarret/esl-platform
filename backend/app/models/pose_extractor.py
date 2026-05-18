"""
Pose Extractor
==============
MediaPipe Holistic — extracts body, hand, and face landmarks from video/images.

Outputs:
  - 33 body pose landmarks
  - 21 left hand landmarks
  - 21 right hand landmarks
  - 468 face mesh landmarks

Saves pose sequences as structured JSON for motion generation.
"""
from __future__ import annotations

import json
import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from app.core.logging import get_logger

logger = get_logger(__name__)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float = 1.0


@dataclass
class PoseFrame:
    frame_idx: int
    timestamp_ms: float
    body: list[Landmark] = field(default_factory=list)       # 33 landmarks
    left_hand: list[Landmark] = field(default_factory=list)  # 21 landmarks
    right_hand: list[Landmark] = field(default_factory=list) # 21 landmarks
    face: list[Landmark] = field(default_factory=list)       # 468 landmarks


@dataclass
class PoseSequence:
    source: str
    fps: float
    total_frames: int
    frames: list[PoseFrame] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, path: Path) -> None:
        path.write_text(self.to_json(), encoding="utf-8")
        logger.info("pose_sequence_saved", path=str(path), frames=len(self.frames))

    @classmethod
    def load(cls, path: Path) -> "PoseSequence":
        data = json.loads(path.read_text(encoding="utf-8"))
        frames = [
            PoseFrame(
                frame_idx=f["frame_idx"],
                timestamp_ms=f["timestamp_ms"],
                body=[Landmark(**lm) for lm in f["body"]],
                left_hand=[Landmark(**lm) for lm in f["left_hand"]],
                right_hand=[Landmark(**lm) for lm in f["right_hand"]],
                face=[Landmark(**lm) for lm in f["face"]],
            )
            for f in data["frames"]
        ]
        return cls(
            source=data["source"],
            fps=data["fps"],
            total_frames=data["total_frames"],
            frames=frames,
        )


def _mp_landmarks_to_list(landmarks, default_count: int) -> list[Landmark]:
    if landmarks is None:
        return [Landmark(0.0, 0.0, 0.0, 0.0)] * default_count
    return [
        Landmark(
            x=lm.x, y=lm.y, z=lm.z,
            visibility=getattr(lm, "visibility", 1.0),
        )
        for lm in landmarks.landmark
    ]


class PoseExtractor:
    """
    Extracts full-body pose sequences from video files or image frames.

    Usage:
        extractor = PoseExtractor()
        sequence = extractor.extract_from_video("signer.mp4")
        sequence.save(Path("pose.json"))
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._min_det = min_detection_confidence
        self._min_track = min_tracking_confidence

    def extract_from_video(self, video_path: str | Path) -> PoseSequence:
        """Extract pose sequence from a video file."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames: list[PoseFrame] = []

        logger.info("extracting_pose", source=str(video_path), fps=fps, total_frames=total)

        with mp_holistic.Holistic(
            min_detection_confidence=self._min_det,
            min_tracking_confidence=self._min_track,
        ) as holistic:
            idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = holistic.process(rgb)
                frames.append(
                    PoseFrame(
                        frame_idx=idx,
                        timestamp_ms=(idx / fps) * 1000,
                        body=_mp_landmarks_to_list(result.pose_landmarks, 33),
                        left_hand=_mp_landmarks_to_list(result.left_hand_landmarks, 21),
                        right_hand=_mp_landmarks_to_list(result.right_hand_landmarks, 21),
                        face=_mp_landmarks_to_list(result.face_landmarks, 468),
                    )
                )
                idx += 1

        cap.release()
        logger.info("pose_extraction_complete", frames=len(frames))
        return PoseSequence(
            source=str(video_path),
            fps=fps,
            total_frames=len(frames),
            frames=frames,
        )

    def extract_from_frame(self, frame: np.ndarray, frame_idx: int = 0) -> PoseFrame:
        """Extract pose from a single numpy image frame (BGR)."""
        with mp_holistic.Holistic(
            min_detection_confidence=self._min_det,
            min_tracking_confidence=self._min_track,
        ) as holistic:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)
            return PoseFrame(
                frame_idx=frame_idx,
                timestamp_ms=0.0,
                body=_mp_landmarks_to_list(result.pose_landmarks, 33),
                left_hand=_mp_landmarks_to_list(result.left_hand_landmarks, 21),
                right_hand=_mp_landmarks_to_list(result.right_hand_landmarks, 21),
                face=_mp_landmarks_to_list(result.face_landmarks, 468),
            )
