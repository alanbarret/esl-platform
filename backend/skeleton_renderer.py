"""
Re-render skeleton videos, trimming sections where no hands are visible.
Keeps only frames where at least one hand has landmarks.
"""
import cv2, numpy as np, json, os, subprocess
from pathlib import Path

MOCAP_DIR   = Path('/root/.openclaw/workspace/esl-platform/data/processed/mocap')
VIDEOS_DIR  = Path('/root/.openclaw/workspace/esl-platform/data/skeleton_videos')

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
]
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

W, H = 640, 360

def render_and_trim(sign: str):
    src = MOCAP_DIR / f"{sign}.json"
    dst = VIDEOS_DIR / f"{sign}.mp4"
    if not src.exists():
        return

    with open(src) as f:
        data = json.load(f)

    fps = data.get('fps', 25)
    frames = data['frames']

    # ── Pass 1: find frames with hands ───────────────────────────────────────
    # Also find longest contiguous segment with hands visible
    has_hand = []
    for fd in frames:
        rh = fd.get('rhand')
        lh = fd.get('lhand')
        # Check if wrist landmark has valid position (not all zeros)
        def valid(h):
            if not h: return False
            w = h[0]
            return not (abs(w[0]) < 0.001 and abs(w[1]) < 0.001)
        has_hand.append(valid(rh) or valid(lh))

    # Expand: include ±5 frame buffer around hand segments for smooth transitions
    BUFFER = 5
    expanded = list(has_hand)
    for i, v in enumerate(has_hand):
        if v:
            for j in range(max(0, i-BUFFER), min(len(has_hand), i+BUFFER+1)):
                expanded[j] = True

    keep = [i for i, v in enumerate(expanded) if v]
    if not keep:
        # No hands detected at all — keep all frames
        keep = list(range(len(frames)))

    kept = len(keep)
    total = len(frames)
    pct = int(kept/total*100)
    print(f"  {sign}: {total}fr → {kept}fr ({pct}% kept)", end='')

    # ── Pass 2: render kept frames ────────────────────────────────────────────
    tmp = f'/tmp/{sign}_trim.avi'
    out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'MJPG'), fps, (W, H))

    for fi in keep:
        fd = frames[fi]
        img = np.zeros((H, W, 3), dtype=np.uint8)
        for y in range(H):
            t = y / H
            img[y, :] = (int(8+t*18), int(8+t*14), int(18+t*32))

        # The mocap JSON may store pose as either image-normalized (0..1) or
        # world-space (meters, centered at hips). Detect by value range.
        pose  = fd.get('pose_img') or fd.get('pose')
        rhand = fd.get('rhand') or fd.get('rh_img') or fd.get('rh')
        lhand = fd.get('lhand') or fd.get('lh_img') or fd.get('lh')

        # If pose is world-space (meters), values fall in roughly -1..+1.
        # Project to image coords using the wrist landmarks of the hands as anchors,
        # OR fall back to a centered isotropic scale.
        pose_is_world = False
        if pose and len(pose) > 0:
            sample = pose[0]
            if -2.0 < sample[0] < 2.0 and -2.0 < sample[1] < 2.0 and (sample[0] < 0 or sample[1] < 0):
                pose_is_world = True

        if pose_is_world and pose:
            # Compute scale so the pose roughly fills the frame vertically.
            ys = [p[1] for p in pose if (len(p) < 4 or p[3] > 0.3)]
            xs = [p[0] for p in pose if (len(p) < 4 or p[3] > 0.3)]
            if ys and xs:
                y_range = max(ys) - min(ys)
                y_scale = (H * 0.7) / y_range if y_range > 0.1 else H
                # Centre horizontally
                x_center = (max(xs) + min(xs)) / 2
                y_top = min(ys)
                def to_px(lm):
                    # pose_world has +Y down; image has +Y down too
                    px = int(W/2 + (lm[0] - x_center) * y_scale)
                    py = int(H * 0.15 + (lm[1] - y_top) * y_scale)
                    return px, py
            else:
                def to_px(lm): return int(lm[0] * W), int(lm[1] * H)
        else:
            def to_px(lm): return int(lm[0] * W), int(lm[1] * H)

        if pose:
            for a, b in POSE_CONNECTIONS:
                if a < len(pose) and b < len(pose):
                    vis_a = pose[a][3] if len(pose[a]) > 3 else 1
                    vis_b = pose[b][3] if len(pose[b]) > 3 else 1
                    if vis_a > 0.3 and vis_b > 0.3:
                        cv2.line(img, to_px(pose[a]), to_px(pose[b]), (80,60,160), 2, cv2.LINE_AA)
                        cv2.line(img, to_px(pose[a]), to_px(pose[b]), (124,58,237), 1, cv2.LINE_AA)
            for lm in pose:
                vis = lm[3] if len(lm) > 3 else 1
                if vis > 0.3:
                    cv2.circle(img, to_px(lm), 4, (80,60,160), -1, cv2.LINE_AA)
                    cv2.circle(img, to_px(lm), 3, (160,130,255), -1, cv2.LINE_AA)

        # Draw hands, anchored at the pose's wrist position when available so the
        # hand isn't floating in the image-normalized coordinate frame.
        # MediaPipe pose wrist indices: 15 = left, 16 = right.
        def draw_hand(hand_lm, wrist_idx, line_dark, line_light, dot_color):
            if not hand_lm: return
            if fi == keep[0]:  # debug first kept frame
                print(f'  draw_hand wrist_idx={wrist_idx} pose_is_world={pose_is_world}', flush=True)
            # If pose data is available, anchor hand at pose wrist & scale by
            # forearm length (elbow → wrist distance in pose pixels).
            anchor = None
            scale = 1.0
            if pose and wrist_idx < len(pose):
                wrist_px = to_px(pose[wrist_idx])
                elbow_idx = 13 if wrist_idx == 15 else 14
                if elbow_idx < len(pose):
                    elbow_px = to_px(pose[elbow_idx])
                    forearm_len = ((wrist_px[0]-elbow_px[0])**2 + (wrist_px[1]-elbow_px[1])**2) ** 0.5
                    if forearm_len > 5:
                        # Hand should be ~70% of forearm length
                        anchor = wrist_px
                        # MediaPipe hand landmark 0 is the wrist. Scale relative to it.
                        hand_size_norm = ((hand_lm[12][0]-hand_lm[0][0])**2 + (hand_lm[12][1]-hand_lm[0][1])**2) ** 0.5
                        if hand_size_norm > 0.01:
                            scale = (forearm_len * 0.7) / (hand_size_norm * max(W, H))

            def hand_to_px(lm):
                if anchor is None:
                    return int(lm[0] * W), int(lm[1] * H)
                # Offset from hand-landmark 0 (wrist), scaled, anchored at pose-wrist
                dx = (lm[0] - hand_lm[0][0]) * W * scale
                dy = (lm[1] - hand_lm[0][1]) * H * scale
                return int(anchor[0] + dx), int(anchor[1] + dy)

            for a, b in HAND_CONNECTIONS:
                if a < len(hand_lm) and b < len(hand_lm):
                    cv2.line(img, hand_to_px(hand_lm[a]), hand_to_px(hand_lm[b]), line_dark, 2, cv2.LINE_AA)
                    cv2.line(img, hand_to_px(hand_lm[a]), hand_to_px(hand_lm[b]), line_light, 1, cv2.LINE_AA)
            for lm in hand_lm:
                cv2.circle(img, hand_to_px(lm), 3, dot_color, -1, cv2.LINE_AA)

        draw_hand(rhand, 16, (160,100,40), (255,165,75), (255,165,75))
        draw_hand(lhand, 15, (80,160,40),  (168,255,75), (168,255,75))

        out.write(img)

    out.release()

    subprocess.run([
        'ffmpeg', '-y', '-i', tmp,
        '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-pix_fmt', 'yuv420p',
        str(dst)
    ], capture_output=True)
    os.unlink(tmp)
    print(f" → {dst.stat().st_size//1024}KB")

signs = sorted(p.stem for p in MOCAP_DIR.glob('*.json'))
print(f"Trimming {len(signs)} videos...")
for sign in signs:
    render_and_trim(sign)
print("Done.")
