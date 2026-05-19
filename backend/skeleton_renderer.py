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

        pose  = fd.get('pose')
        rhand = fd.get('rhand')
        lhand = fd.get('lhand')

        def to_px(lm):
            return int(lm[0] * W), int(lm[1] * H)

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

        if rhand:
            for a, b in HAND_CONNECTIONS:
                if a < len(rhand) and b < len(rhand):
                    cv2.line(img, to_px(rhand[a]), to_px(rhand[b]), (160,100,40), 2, cv2.LINE_AA)
                    cv2.line(img, to_px(rhand[a]), to_px(rhand[b]), (255,165,75), 1, cv2.LINE_AA)
            for lm in rhand:
                cv2.circle(img, to_px(lm), 3, (255,165,75), -1, cv2.LINE_AA)

        if lhand:
            for a, b in HAND_CONNECTIONS:
                if a < len(lhand) and b < len(lhand):
                    cv2.line(img, to_px(lhand[a]), to_px(lhand[b]), (80,160,40), 2, cv2.LINE_AA)
                    cv2.line(img, to_px(lhand[a]), to_px(lhand[b]), (168,255,75), 1, cv2.LINE_AA)
            for lm in lhand:
                cv2.circle(img, to_px(lm), 3, (168,255,75), -1, cv2.LINE_AA)

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
