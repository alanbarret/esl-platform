#!/usr/bin/env python3
"""Overlay MediaPipe landmarks (with labels) on top of the rendered 3D avatar video.

Reads:
  - Pre-extracted holistic JSON (pose, lh, rh per frame in image-normalized coords)
  - The rendered avatar MP4
Writes:
  - A new MP4 with landmarks + labels drawn on top of avatar frames

Strategy:
  We have no direct mapping from avatar world coords -> image coords (avatar is
  rendered by Three.js with its own camera). So we draw the landmarks on a
  SECOND panel next to the avatar (side-by-side compare), with the source frame
  for visual context, and label each point.

Usage:
  python3 overlay_landmarks.py <avatar.mp4> <source_video.mp4> <holistic.json> <out.mp4>
"""

import sys, argparse, json
import cv2
import numpy as np
from pathlib import Path


# MediaPipe Pose landmark names (BlazePose 33)
POSE_LABELS = {
    0: 'nose', 1: 'l_eye_in', 2: 'l_eye', 3: 'l_eye_out',
    4: 'r_eye_in', 5: 'r_eye', 6: 'r_eye_out',
    7: 'l_ear', 8: 'r_ear',
    9: 'mouth_l', 10: 'mouth_r',
    11: 'l_shoulder', 12: 'r_shoulder',
    13: 'l_elbow', 14: 'r_elbow',
    15: 'l_wrist', 16: 'r_wrist',
    17: 'l_pinky_p', 18: 'r_pinky_p',
    19: 'l_index_p', 20: 'r_index_p',
    21: 'l_thumb_p', 22: 'r_thumb_p',
    23: 'l_hip', 24: 'r_hip',
    25: 'l_knee', 26: 'r_knee',
    27: 'l_ankle', 28: 'r_ankle',
    29: 'l_heel', 30: 'r_heel',
    31: 'l_foot_idx', 32: 'r_foot_idx',
}

HAND_LABELS = {
    0: 'wrist',
    1: 'th1', 2: 'th2', 3: 'th3', 4: 'th4',
    5: 'in1', 6: 'in2', 7: 'in3', 8: 'in4',
    9: 'mi1', 10: 'mi2', 11: 'mi3', 12: 'mi4',
    13: 'ri1', 14: 'ri2', 15: 'ri3', 16: 'ri4',
    17: 'pi1', 18: 'pi2', 19: 'pi3', 20: 'pi4',
}

# Pose skeleton edges (which landmark pairs to connect with lines)
POSE_EDGES = [
    (11,12), (11,13), (13,15), (12,14), (14,16),  # arms + shoulders
    (11,23), (12,24), (23,24),  # torso
    (23,25), (25,27), (24,26), (26,28),  # legs
    (15,17), (15,19), (15,21), (17,19),  # left palm sketch
    (16,18), (16,20), (16,22), (18,20),  # right palm sketch
]
HAND_EDGES = [
    # Thumb
    (0,1),(1,2),(2,3),(3,4),
    # Index
    (0,5),(5,6),(6,7),(7,8),
    # Middle
    (0,9),(9,10),(10,11),(11,12),
    # Ring
    (0,13),(13,14),(14,15),(15,16),
    # Pinky
    (0,17),(17,18),(18,19),(19,20),
    # Palm
    (5,9),(9,13),(13,17),
]


# Avatar bone-edge graph (which bone connects to which for line drawing)
AVATAR_EDGES = [
    ('Hips', 'Spine'), ('Spine', 'Spine2'), ('Spine2', 'Neck'), ('Neck', 'Head'),
    ('Spine2', 'LeftShoulder'), ('LeftShoulder', 'LeftArm'),
    ('LeftArm', 'LeftForeArm'), ('LeftForeArm', 'LeftHand'),
    ('Spine2', 'RightShoulder'), ('RightShoulder', 'RightArm'),
    ('RightArm', 'RightForeArm'), ('RightForeArm', 'RightHand'),
    # Finger chains
    ('LeftHand', 'LeftHandThumb1'), ('LeftHandThumb1', 'LeftHandThumb2'), ('LeftHandThumb2', 'LeftHandThumb3'),
    ('LeftHand', 'LeftHandIndex1'), ('LeftHandIndex1', 'LeftHandIndex2'), ('LeftHandIndex2', 'LeftHandIndex3'),
    ('LeftHand', 'LeftHandMiddle1'), ('LeftHandMiddle1', 'LeftHandMiddle2'), ('LeftHandMiddle2', 'LeftHandMiddle3'),
    ('LeftHand', 'LeftHandRing1'), ('LeftHandRing1', 'LeftHandRing2'), ('LeftHandRing2', 'LeftHandRing3'),
    ('LeftHand', 'LeftHandPinky1'), ('LeftHandPinky1', 'LeftHandPinky2'), ('LeftHandPinky2', 'LeftHandPinky3'),
    ('RightHand', 'RightHandThumb1'), ('RightHandThumb1', 'RightHandThumb2'), ('RightHandThumb2', 'RightHandThumb3'),
    ('RightHand', 'RightHandIndex1'), ('RightHandIndex1', 'RightHandIndex2'), ('RightHandIndex2', 'RightHandIndex3'),
    ('RightHand', 'RightHandMiddle1'), ('RightHandMiddle1', 'RightHandMiddle2'), ('RightHandMiddle2', 'RightHandMiddle3'),
    ('RightHand', 'RightHandRing1'), ('RightHandRing1', 'RightHandRing2'), ('RightHandRing2', 'RightHandRing3'),
    ('RightHand', 'RightHandPinky1'), ('RightHandPinky1', 'RightHandPinky2'), ('RightHandPinky2', 'RightHandPinky3'),
]

# Short labels for avatar bones (compact)
AVATAR_LABELS = {
    'Hips': 'hip', 'Spine': 'spine', 'Spine2': 'sp2', 'Neck': 'neck', 'Head': 'head',
    'LeftShoulder': 'L_sh', 'LeftArm': 'L_arm', 'LeftForeArm': 'L_fa', 'LeftHand': 'L_wr',
    'RightShoulder': 'R_sh', 'RightArm': 'R_arm', 'RightForeArm': 'R_fa', 'RightHand': 'R_wr',
    'LeftHandThumb1': 'th1', 'LeftHandThumb2': 'th2', 'LeftHandThumb3': 'th3',
    'LeftHandIndex1': 'in1', 'LeftHandIndex2': 'in2', 'LeftHandIndex3': 'in3',
    'LeftHandMiddle1': 'mi1', 'LeftHandMiddle2': 'mi2', 'LeftHandMiddle3': 'mi3',
    'LeftHandRing1': 'ri1', 'LeftHandRing2': 'ri2', 'LeftHandRing3': 'ri3',
    'LeftHandPinky1': 'pi1', 'LeftHandPinky2': 'pi2', 'LeftHandPinky3': 'pi3',
    'RightHandThumb1': 'th1', 'RightHandThumb2': 'th2', 'RightHandThumb3': 'th3',
    'RightHandIndex1': 'in1', 'RightHandIndex2': 'in2', 'RightHandIndex3': 'in3',
    'RightHandMiddle1': 'mi1', 'RightHandMiddle2': 'mi2', 'RightHandMiddle3': 'mi3',
    'RightHandRing1': 'ri1', 'RightHandRing2': 'ri2', 'RightHandRing3': 'ri3',
    'RightHandPinky1': 'pi1', 'RightHandPinky2': 'pi2', 'RightHandPinky3': 'pi3',
}


def draw_source_overlay_on_avatar(canvas, frame_data, bones, panel_x, panel_y, panel_w, panel_h):
    """Overlay source landmarks (from MediaPipe) onto the avatar panel, aligned to the
    avatar's body using the SHOULDER MIDPOINT as the anchor and SHOULDER WIDTH as scale.
    Draws in semi-transparent red so it's distinguishable from the avatar bones."""
    pose_img = frame_data.get('pose_img')
    if not pose_img or 'LeftShoulder' not in bones or 'RightShoulder' not in bones:
        return

    pose_arr = np.array(pose_img)
    # Source anchor: midpoint of pose shoulders (image-normalized 0..1).
    src_ls = pose_arr[11, :2]; src_rs = pose_arr[12, :2]
    src_mid = (src_ls + src_rs) * 0.5
    src_sh_w = float(np.linalg.norm(src_ls - src_rs))
    if src_sh_w < 1e-4: return

    # Avatar anchor: midpoint of avatar shoulders (in avatar panel NDC 0..1).
    av_ls = np.array(bones['LeftShoulder'][:2])
    av_rs = np.array(bones['RightShoulder'][:2])
    av_mid = (av_ls + av_rs) * 0.5
    av_sh_w = float(np.linalg.norm(av_ls - av_rs))
    if av_sh_w < 1e-4: return

    scale = av_sh_w / src_sh_w

    def remap(src_xy):
        # Translate source point by its offset from src_mid, scaled, then add av_mid.
        rel = (src_xy - src_mid) * scale
        return av_mid + rel

    def to_px(rel_xy):
        return int(panel_x + rel_xy[0] * panel_w), int(panel_y + rel_xy[1] * panel_h)

    # Draw POSE landmarks (skip face)
    for a, b in POSE_EDGES:
        pa = to_px(remap(pose_arr[a, :2]))
        pb = to_px(remap(pose_arr[b, :2]))
        cv2.line(canvas, pa, pb, (60, 60, 220), 1)
    for i in range(33):
        if i in (1,2,3,4,5,6,7,8,9,10,17,18,19,20,21,22,29,30,31,32):
            continue
        p = to_px(remap(pose_arr[i, :2]))
        cv2.circle(canvas, p, 4, (60, 60, 220), -1)
        cv2.circle(canvas, p, 4, (255, 255, 255), 1)

    # Hands too
    for hand_key, col in [('lh', (60, 200, 60)), ('rh', (220, 140, 60))]:
        hand = frame_data.get(hand_key)
        if not hand: continue
        h_arr = np.array(hand)
        # Remap each hand landmark using the same shoulder-anchored scale.
        pts = [to_px(remap(h_arr[k, :2])) for k in range(len(h_arr))]
        for a, b in HAND_EDGES:
            cv2.line(canvas, pts[a], pts[b], col, 1)
        for pt in pts:
            cv2.circle(canvas, pt, 3, col, -1)

    # Legend
    cv2.putText(canvas, 'SRC (red=pose, green=L-hand, orange=R-hand)',
                (panel_x + 10, panel_y + panel_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)


def draw_bones_on_avatar(canvas, bones, panel_x, panel_y, panel_w, panel_h):
    """Draw skeleton + labels on the avatar panel using the bones dict.
    bones[name] = [x_ndc_norm, y_ndc_norm, z_ndc] where x/y are in 0..1."""
    def to_px(b):
        x, y, _ = b
        return int(panel_x + x * panel_w), int(panel_y + y * panel_h)

    # Edges
    for a, b in AVATAR_EDGES:
        if a in bones and b in bones:
            cv2.line(canvas, to_px(bones[a]), to_px(bones[b]), (255, 200, 100), 1)

    # Color helpers
    def color_for(name):
        if 'LeftHand' in name and len(name) > 8: return (100, 255, 100)
        if 'RightHand' in name and len(name) > 9: return (100, 200, 255)
        return (255, 200, 100)

    for name, b in bones.items():
        p = to_px(b)
        col = color_for(name)
        cv2.circle(canvas, p, 3, col, -1)
        cv2.circle(canvas, p, 3, (0, 0, 0), 1)
        if name in AVATAR_LABELS:
            cv2.putText(canvas, AVATAR_LABELS[name], (p[0]+4, p[1]-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)


def draw_landmarks_panel(canvas, frame_data, src_w, src_h, panel_x, panel_y, panel_w, panel_h, bg_img=None):
    """Draw pose + hand landmarks on a (panel_w x panel_h) sub-canvas."""
    # Optional background (source video frame)
    if bg_img is not None:
        bg = cv2.resize(bg_img, (panel_w, panel_h))
        canvas[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w] = bg
    else:
        canvas[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w] = (40, 40, 50)

    if not frame_data:
        return

    def to_px(x, y):
        return int(panel_x + x * panel_w), int(panel_y + y * panel_h)

    # ---- Pose (image-normalized, aligns with source image) ----
    pose_img = frame_data.get('pose_img')
    if pose_img:
        arr = np.array(pose_img)
        def proj(lm):
            return to_px(lm[0], lm[1])
        # Draw edges first
        for a, b in POSE_EDGES:
            pa = proj(arr[a]); pb = proj(arr[b])
            cv2.line(canvas, pa, pb, (180, 180, 255), 1)
        # Then points + labels
        for i in range(33):
            if i in (1,2,3,4,5,6,7,8,9,10):  # skip face/eye/ear for clarity
                continue
            p = proj(arr[i])
            cv2.circle(canvas, p, 3, (255, 255, 100), -1)
            cv2.circle(canvas, p, 3, (0, 0, 0), 1)
            if i in POSE_LABELS:
                label = POSE_LABELS[i]
                cv2.putText(canvas, label, (p[0]+4, p[1]-3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    elif frame_data.get('pose'):
        # Fallback: world coords (old JSON without pose_img). Approximate projection.
        arr = np.array(frame_data['pose'])[:, :2]
        cx, cy = (arr[23, 0] + arr[24, 0]) * 0.5, (arr[23, 1] + arr[24, 1]) * 0.5
        shoulder_w = abs(arr[11, 0] - arr[12, 0]) or 0.3
        scale_x = panel_w * 0.25 / shoulder_w
        scale_y = scale_x
        offset_x = panel_w * 0.5
        offset_y = panel_h * 0.4
        def proj(lm):
            x = (lm[0] - cx) * scale_x + offset_x
            y = (lm[1] - cy) * scale_y + offset_y
            return int(panel_x + x), int(panel_y + y)
        for a, b in POSE_EDGES:
            cv2.line(canvas, proj(arr[a]), proj(arr[b]), (180, 180, 255), 1)
        for i in range(33):
            if i in (1,2,3,4,5,6,7,8,9,10):
                continue
            p = proj(arr[i])
            cv2.circle(canvas, p, 3, (255, 255, 100), -1)
            if i in POSE_LABELS:
                cv2.putText(canvas, POSE_LABELS[i], (p[0]+4, p[1]-3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

    # ---- Hands (use image-normalized coords directly) ----
    for hand_key, color in [('lh', (100, 255, 100)), ('rh', (100, 200, 255))]:
        hand = frame_data.get(hand_key)
        if not hand:
            continue
        hand_arr = np.array(hand)
        # Image-normalized 0..1
        pts = [to_px(p[0], p[1]) for p in hand_arr]
        for a, b in HAND_EDGES:
            cv2.line(canvas, pts[a], pts[b], color, 1)
        for i, pt in enumerate(pts):
            cv2.circle(canvas, pt, 3, color, -1)
            cv2.circle(canvas, pt, 3, (0, 0, 0), 1)
            label = HAND_LABELS.get(i, str(i))
            cv2.putText(canvas, label, (pt[0]+4, pt[1]+3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('avatar_mp4')
    ap.add_argument('source_mp4')
    ap.add_argument('holistic_json')
    ap.add_argument('out_mp4')
    ap.add_argument('--trim-start', type=int, default=0, help='Trim N leading frames from source/holistic (must match retarget trimming)')
    ap.add_argument('--trim-end', type=int, default=None)
    ap.add_argument('--bones-json', default=None, help='Optional bones JSON from render.js for avatar overlay')
    args = ap.parse_args()

    bones_data = None
    if args.bones_json and Path(args.bones_json).exists():
        with open(args.bones_json) as f:
            bones_data = json.load(f)
        print(f"Loaded bones JSON: {len(bones_data['frames'])} frames")

    with open(args.holistic_json) as f:
        d = json.load(f)
    frames_data = d['frames']

    cap_av = cv2.VideoCapture(args.avatar_mp4)
    cap_src = cv2.VideoCapture(args.source_mp4)
    av_w = int(cap_av.get(cv2.CAP_PROP_FRAME_WIDTH))
    av_h = int(cap_av.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_w = int(cap_src.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap_src.get(cv2.CAP_PROP_FRAME_HEIGHT))
    av_fps = cap_av.get(cv2.CAP_PROP_FPS) or 25.0
    n_av = int(cap_av.get(cv2.CAP_PROP_FRAME_COUNT))

    # Three panels: source + landmarks-overlay + avatar
    panel_w = av_w
    panel_h = av_h
    out_w = panel_w * 3
    out_h = panel_h

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path_tmp = str(Path(args.out_mp4).with_suffix('.tmp.mp4'))
    writer = cv2.VideoWriter(out_path_tmp, fourcc, av_fps, (out_w, out_h))

    # Trim
    trim_start = args.trim_start
    trim_end = args.trim_end if args.trim_end is not None else len(frames_data)
    frames_data = frames_data[trim_start:trim_end]
    print(f"Avatar frames: {n_av}, Holistic frames after trim: {len(frames_data)}")

    # Loop over avatar frames; sync to holistic via index ratio
    i = 0
    while True:
        ok_av, av_img = cap_av.read()
        if not ok_av: break
        # Map avatar frame index -> holistic frame index
        if len(frames_data) <= 1:
            hi = 0
        else:
            hi = min(len(frames_data) - 1, int(i * (len(frames_data) - 1) / max(1, n_av - 1)))
        fdata = frames_data[hi]

        # Get matching source frame
        cap_src.set(cv2.CAP_PROP_POS_FRAMES, trim_start + hi)
        ok_src, src_img = cap_src.read()
        if not ok_src or src_img is None:
            src_img = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        else:
            src_img = cv2.resize(src_img, (panel_w, panel_h))

        # Build canvas: [source | source-with-landmarks | avatar]
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        # Panel 1: source
        canvas[:, 0:panel_w] = src_img
        cv2.putText(canvas, 'SOURCE', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Panel 2: source with landmarks
        draw_landmarks_panel(canvas, fdata, src_w, src_h, panel_w, 0, panel_w, panel_h, bg_img=src_img)
        cv2.putText(canvas, 'LANDMARKS', (panel_w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Panel 3: avatar
        canvas[:, 2*panel_w:3*panel_w] = av_img
        cv2.putText(canvas, 'AVATAR', (2*panel_w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Avatar bone overlay (if available)
        if bones_data and i < len(bones_data['frames']) and bones_data['frames'][i]:
            bones = bones_data['frames'][i]
            draw_bones_on_avatar(canvas, bones, 2*panel_w, 0, panel_w, panel_h)
            # Also overlay the SOURCE landmarks on the avatar panel, aligned to the
            # avatar's body using shoulder midpoints as anchor and shoulder width as scale.
            draw_source_overlay_on_avatar(canvas, fdata, bones, 2*panel_w, 0, panel_w, panel_h)

        # Frame counter
        cv2.putText(canvas, f"f{i} (h{hi})", (out_w - 100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        writer.write(canvas)
        i += 1

    cap_av.release(); cap_src.release(); writer.release()

    # Re-encode with libx264 for compatibility
    import subprocess
    subprocess.run([
        'ffmpeg', '-y', '-i', out_path_tmp,
        '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p',
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        args.out_mp4,
    ], check=True)
    Path(out_path_tmp).unlink(missing_ok=True)
    print(f"✅ Wrote {args.out_mp4}")


if __name__ == '__main__':
    main()
