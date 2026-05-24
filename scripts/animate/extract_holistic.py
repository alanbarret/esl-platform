#!/usr/bin/env python3
"""Run MediaPipe Holistic on a source video and save body + hand landmarks.

Output JSON:
{
  "fps": float,
  "frames": [
    {
      "pose": [[x,y,z,vis], ... 33],   # MediaPipe pose_world_landmarks
      "lh":   [[x,y,z], ... 21] | null, # MediaPipe left_hand_landmarks (image-relative)
      "rh":   [[x,y,z], ... 21] | null
    },
    ...
  ]
}
"""
import sys, json, argparse
import cv2
import numpy as np
import mediapipe as mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('output_json')
    ap.add_argument('--max-frames', type=int, default=None)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {args.video} | {fps:.1f}fps | {n_total} frames")

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
    )

    frames = []
    i = 0
    while True:
        ok, img = cap.read()
        if not ok: break
        if args.max_frames and i >= args.max_frames: break
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = holistic.process(rgb)
        f = {'pose': None, 'pose_img': None, 'lh': None, 'rh': None}
        if res.pose_world_landmarks:
            f['pose'] = [[lm.x, lm.y, lm.z, lm.visibility] for lm in res.pose_world_landmarks.landmark]
        if res.pose_landmarks:
            f['pose_img'] = [[lm.x, lm.y, lm.z, lm.visibility] for lm in res.pose_landmarks.landmark]
        if res.left_hand_landmarks:
            f['lh'] = [[lm.x, lm.y, lm.z] for lm in res.left_hand_landmarks.landmark]
        if res.right_hand_landmarks:
            f['rh'] = [[lm.x, lm.y, lm.z] for lm in res.right_hand_landmarks.landmark]
        frames.append(f)
        i += 1
        if i % 30 == 0:
            print(f"  ...{i} frames")

    cap.release(); holistic.close()
    out = {'fps': fps, 'frames': frames, 'world_landmarks': True, 'has_hands': True}
    with open(args.output_json, 'w') as f:
        json.dump(out, f)
    print(f"Wrote {args.output_json} ({len(frames)} frames)")
    n_pose = sum(1 for f in frames if f['pose'])
    n_lh = sum(1 for f in frames if f['lh'])
    n_rh = sum(1 for f in frames if f['rh'])
    print(f"  Pose detected: {n_pose}/{len(frames)}")
    print(f"  Left hand:     {n_lh}/{len(frames)}")
    print(f"  Right hand:    {n_rh}/{len(frames)}")


if __name__ == '__main__':
    main()
