#!/usr/bin/env python3
"""
Compare the avatar's actual bone positions (from bones.json) against the source
MediaPipe landmarks (from holistic JSON), to identify where the retargeter
diverges from the source pose.

Strategy:
  - Both are normalized to image-space [0..1] in their respective frames.
  - Align them using shoulder midpoint + shoulder width (same as overlay viz).
  - Compute per-bone Euclidean error in normalized units (scaled to "shoulder
    widths") for each frame.
  - Identify which bones diverge the most and in which directions.

Usage:
  python3 diagnose_errors.py <bones.json> <holistic.json> [--trim-start N]
"""

import sys, json, argparse
import numpy as np


# Map: pose landmark index -> avatar bone name (semantic correspondence)
POSE_TO_BONE = {
    11: 'LeftShoulder',
    12: 'RightShoulder',
    13: 'LeftArm',          # elbow ↔ end of upper arm
    14: 'RightArm',
    15: 'LeftForeArm',      # wrist ↔ end of forearm
    16: 'RightForeArm',
}

# Hand landmark index -> avatar bone name
HAND_TO_BONE = {
    'Left': {
        1: 'LeftHandThumb1', 2: 'LeftHandThumb2', 3: 'LeftHandThumb3',
        5: 'LeftHandIndex1', 6: 'LeftHandIndex2', 7: 'LeftHandIndex3',
        9: 'LeftHandMiddle1', 10: 'LeftHandMiddle2', 11: 'LeftHandMiddle3',
        13: 'LeftHandRing1', 14: 'LeftHandRing2', 15: 'LeftHandRing3',
        17: 'LeftHandPinky1', 18: 'LeftHandPinky2', 19: 'LeftHandPinky3',
    },
    'Right': {
        1: 'RightHandThumb1', 2: 'RightHandThumb2', 3: 'RightHandThumb3',
        5: 'RightHandIndex1', 6: 'RightHandIndex2', 7: 'RightHandIndex3',
        9: 'RightHandMiddle1', 10: 'RightHandMiddle2', 11: 'RightHandMiddle3',
        13: 'RightHandRing1', 14: 'RightHandRing2', 15: 'RightHandRing3',
        17: 'RightHandPinky1', 18: 'RightHandPinky2', 19: 'RightHandPinky3',
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bones_json')
    ap.add_argument('holistic_json')
    ap.add_argument('--trim-start', type=int, default=0)
    args = ap.parse_args()

    with open(args.bones_json) as f:
        b = json.load(f)
    with open(args.holistic_json) as f:
        h = json.load(f)

    avatar_frames = b['frames']
    src_frames = h['frames'][args.trim_start:args.trim_start + len(avatar_frames)]
    print(f"Comparing {len(avatar_frames)} avatar frames vs {len(src_frames)} source frames")

    # Aggregate errors per bone
    bone_errors = {}  # bone -> list of (frame, dx, dy, mag_in_shoulder_widths)

    for fi, (av, src) in enumerate(zip(avatar_frames, src_frames)):
        if not av: continue
        # Need shoulders in both
        if 'LeftShoulder' not in av or 'RightShoulder' not in av: continue
        if not src.get('pose_img'): continue
        pose = np.array(src['pose_img'])
        av_ls = np.array(av['LeftShoulder'][:2]); av_rs = np.array(av['RightShoulder'][:2])
        av_mid = (av_ls + av_rs) * 0.5
        av_sh_w = float(np.linalg.norm(av_ls - av_rs))
        if av_sh_w < 1e-4: continue

        src_ls = pose[11, :2]; src_rs = pose[12, :2]
        src_mid = (src_ls + src_rs) * 0.5
        src_sh_w = float(np.linalg.norm(src_ls - src_rs))
        if src_sh_w < 1e-4: continue

        scale = av_sh_w / src_sh_w

        def src_to_avatar_space(src_xy):
            return av_mid + (src_xy - src_mid) * scale

        # ---- Pose-based bone comparisons ----
        for pose_idx, bone in POSE_TO_BONE.items():
            if bone not in av: continue
            src_pos = src_to_avatar_space(pose[pose_idx, :2])
            av_pos = np.array(av[bone][:2])
            err = src_pos - av_pos
            mag = float(np.linalg.norm(err)) / av_sh_w
            bone_errors.setdefault(bone, []).append((fi, err[0], err[1], mag))

        # ---- Hand-based bone comparisons ----
        for side, key in [('Left', 'lh'), ('Right', 'rh')]:
            hand_lm = src.get(key)
            if not hand_lm: continue
            harr = np.array(hand_lm)
            # The hand landmarks share the same image coordinate space as pose_img,
            # so the same scale/anchor remap works.
            for h_idx, bone in HAND_TO_BONE[side].items():
                if bone not in av: continue
                src_pos = src_to_avatar_space(harr[h_idx, :2])
                av_pos = np.array(av[bone][:2])
                err = src_pos - av_pos
                mag = float(np.linalg.norm(err)) / av_sh_w
                bone_errors.setdefault(bone, []).append((fi, err[0], err[1], mag))

    # Summarize
    print(f"\n{'Bone':24s} {'frames':>6s} {'mean_dx':>10s} {'mean_dy':>10s} {'mean_mag':>10s} {'max_mag':>10s}")
    print("-" * 80)
    rows = []
    for bone, errs in bone_errors.items():
        if not errs: continue
        errs_np = np.array([[e[1], e[2], e[3]] for e in errs])
        mean_dx = float(errs_np[:, 0].mean())
        mean_dy = float(errs_np[:, 1].mean())
        mean_mag = float(errs_np[:, 2].mean())
        max_mag = float(errs_np[:, 2].max())
        rows.append((bone, len(errs), mean_dx, mean_dy, mean_mag, max_mag))

    # Sort by mean magnitude (worst first)
    rows.sort(key=lambda r: -r[4])
    for r in rows:
        print(f"{r[0]:24s} {r[1]:>6d} {r[2]:>+10.3f} {r[3]:>+10.3f} {r[4]:>10.3f} {r[5]:>10.3f}")

    print()
    print("Errors are in SHOULDER WIDTHS (so 1.0 = one shoulder-width off).")
    print("mean_dx > 0 means avatar bone is to the LEFT of where source says it should be")
    print("mean_dy > 0 means avatar bone is ABOVE where source says it should be")
    print("(image y increases downward; dy>0 = avatar higher in image = above)")

    # Pull out interesting summary stats
    print()
    print("=" * 80)
    print("SUMMARY:")
    upper_body = [r for r in rows if r[0] in {'LeftShoulder','RightShoulder','LeftArm','RightArm','LeftForeArm','RightForeArm'}]
    if upper_body:
        avg_mag = np.mean([r[4] for r in upper_body])
        print(f"  Upper body (shoulders/arms) avg error: {avg_mag:.3f} shoulder-widths")
    left_hand = [r for r in rows if r[0].startswith('LeftHand') and r[0] != 'LeftHand']
    right_hand = [r for r in rows if r[0].startswith('RightHand') and r[0] != 'RightHand']
    if left_hand:
        avg = np.mean([r[4] for r in left_hand])
        avg_dx = np.mean([r[2] for r in left_hand])
        avg_dy = np.mean([r[3] for r in left_hand])
        print(f"  Left hand fingers avg error: {avg:.3f} shoulder-widths  bias=({avg_dx:+.3f}, {avg_dy:+.3f})")
    if right_hand:
        avg = np.mean([r[4] for r in right_hand])
        avg_dx = np.mean([r[2] for r in right_hand])
        avg_dy = np.mean([r[3] for r in right_hand])
        print(f"  Right hand fingers avg error: {avg:.3f} shoulder-widths  bias=({avg_dx:+.3f}, {avg_dy:+.3f})")


if __name__ == '__main__':
    main()
