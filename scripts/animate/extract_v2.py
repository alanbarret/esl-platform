#!/usr/bin/env python3
"""
Extract 3D landmarks using MediaPipe's NEW Tasks API (HandLandmarker + PoseLandmarker).

Key difference from legacy mp.solutions.holistic:
- Returns hand_world_landmarks in METRIC METERS (using GHUM 3D human model)
- pose_world_landmarks in metric meters
- More reliable 3D depth (1.3 cm mean error per Google's ASL benchmark)

Output JSON format (compatible with retarget_digihuman.py):
{
  "fps": float,
  "frames": [
    {
      "pose":     [[x, y, z, visibility], ... 33],     # pose_world_landmarks (meters)
      "pose_img": [[x, y, z, visibility], ... 33],     # pose image-normalized (0..1)
      "lh":       [[x, y, z], ... 21] or None,         # left hand world (METERS - reliable!)
      "rh":       [[x, y, z], ... 21] or None,         # right hand world (METERS - reliable!)
      "lh_img":   [[x, y, z], ... 21] or None,         # left hand image-relative
      "rh_img":   [[x, y, z], ... 21] or None,         # right hand image-relative
    },
    ...
  ],
  "world_landmarks": true,
  "has_hands": true
}
"""
import sys
import argparse
import json
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions,
    PoseLandmarker, PoseLandmarkerOptions,
    RunningMode,
)


MODELS_DIR = '/root/.openclaw/workspace/esl-platform/data/mediapipe_models'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('output_json')
    ap.add_argument('--max-frames', type=int, default=None)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {args.video} | {fps:.1f}fps | {n_total} frames")

    # Build PoseLandmarker (video mode)
    pose_opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=f'{MODELS_DIR}/pose_landmarker_full.task'),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    # Build HandLandmarker (video mode, 2 hands)
    hand_opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=f'{MODELS_DIR}/hand_landmarker.task'),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames = []
    pose_lm = PoseLandmarker.create_from_options(pose_opts)
    hand_lm = HandLandmarker.create_from_options(hand_opts)

    try:
        i = 0
        while True:
            ok, img = cap.read()
            if not ok: break
            if args.max_frames and i >= args.max_frames: break
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(i * 1000 / fps)

            pose_res = pose_lm.detect_for_video(mp_image, ts_ms)
            hand_res = hand_lm.detect_for_video(mp_image, ts_ms)

            f = {'pose': None, 'pose_img': None,
                 'lh': None, 'rh': None,
                 'lh_img': None, 'rh_img': None}

            # Pose
            if pose_res.pose_world_landmarks and pose_res.pose_world_landmarks[0]:
                pwl = pose_res.pose_world_landmarks[0]
                f['pose'] = [[lm.x, lm.y, lm.z,
                              lm.visibility if lm.visibility else 1.0] for lm in pwl]
            if pose_res.pose_landmarks and pose_res.pose_landmarks[0]:
                pl = pose_res.pose_landmarks[0]
                f['pose_img'] = [[lm.x, lm.y, lm.z,
                                  lm.visibility if lm.visibility else 1.0] for lm in pl]

            # Hands
            if hand_res.hand_world_landmarks:
                for idx, hlm in enumerate(hand_res.hand_world_landmarks):
                    # handedness tells us which hand
                    handedness = hand_res.handedness[idx][0].category_name  # "Left" or "Right"
                    key = 'lh' if handedness == 'Left' else 'rh'
                    f[key] = [[p.x, p.y, p.z] for p in hlm]
            if hand_res.hand_landmarks:
                for idx, hlm in enumerate(hand_res.hand_landmarks):
                    handedness = hand_res.handedness[idx][0].category_name
                    key = 'lh_img' if handedness == 'Left' else 'rh_img'
                    f[key] = [[p.x, p.y, p.z] for p in hlm]

            frames.append(f)
            i += 1
            if i % 30 == 0:
                print(f"  ...{i} frames")

    finally:
        pose_lm.close()
        hand_lm.close()
        cap.release()

    out = {'fps': fps, 'frames': frames, 'world_landmarks': True, 'has_hands': True,
           'metric_world_hands': True}
    with open(args.output_json, 'w') as f:
        json.dump(out, f)
    print(f"Wrote {args.output_json} ({len(frames)} frames)")
    print(f"  Pose detected:     {sum(1 for f in frames if f['pose'])}/{len(frames)}")
    print(f"  Pose image:        {sum(1 for f in frames if f['pose_img'])}/{len(frames)}")
    print(f"  Left hand world:   {sum(1 for f in frames if f['lh'])}/{len(frames)}")
    print(f"  Right hand world:  {sum(1 for f in frames if f['rh'])}/{len(frames)}")


if __name__ == '__main__':
    main()
