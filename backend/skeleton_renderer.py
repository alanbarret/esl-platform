"""
Renders a skeleton video from pre-extracted mocap landmark JSON.
No MediaPipe needed at render time — uses stored landmark data.
"""
import cv2, numpy as np, json, math, subprocess, os, tempfile
from pathlib import Path

MOCAP_DIR = Path(__file__).parent.parent / "data" / "processed" / "mocap"
VIDEOS_DIR = Path(__file__).parent.parent / "data" / "skeleton_videos"
VIDEOS_DIR.mkdir(exist_ok=True)

# ── MediaPipe connection topology (hardcoded, no mp needed) ──────────────────
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

def render_skeleton_video(sign: str, label: str = None) -> str | None:
    """Render skeleton video for a sign. Returns path to MP4 or None."""
    mocap_path = MOCAP_DIR / f"{sign.upper()}.json"
    if not mocap_path.exists():
        return None

    out_path = VIDEOS_DIR / f"{sign.upper()}.mp4"
    if out_path.exists():
        return str(out_path)

    with open(mocap_path) as f:
        data = json.load(f)

    fps = data.get('fps', 25)
    frames = data['frames']
    W, H = 640, 360
    sign_label = label or sign.replace('_', ' ').title()

    tmp = tempfile.mktemp(suffix='.avi')
    out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'MJPG'), fps, (W, H))
    TOTAL = len(frames)

    for fi, fd in enumerate(frames):
        # Background gradient
        img = np.zeros((H, W, 3), dtype=np.uint8)
        for y in range(H):
            t = y / H
            img[y, :] = (int(8 + t*18), int(8 + t*14), int(18 + t*32))

        pose  = fd.get('pose')
        rhand = fd.get('rhand')
        lhand = fd.get('lhand')

        def to_px(lm):
            return int(lm[0] * W), int(lm[1] * H)

        # Draw pose skeleton
        if pose:
            for a, b in POSE_CONNECTIONS:
                if a < len(pose) and b < len(pose):
                    vis_a = pose[a][3] if len(pose[a]) > 3 else 1
                    vis_b = pose[b][3] if len(pose[b]) > 3 else 1
                    if vis_a > 0.3 and vis_b > 0.3:
                        pa, pb = to_px(pose[a]), to_px(pose[b])
                        cv2.line(img, pa, pb, (80, 60, 160), 2, cv2.LINE_AA)
                        cv2.line(img, pa, pb, (124, 58, 237), 1, cv2.LINE_AA)
            # Draw joints
            for i, lm in enumerate(pose):
                vis = lm[3] if len(lm) > 3 else 1
                if vis > 0.3:
                    px = to_px(lm)
                    cv2.circle(img, px, 4, (80, 60, 160), -1, cv2.LINE_AA)
                    cv2.circle(img, px, 3, (160, 130, 255), -1, cv2.LINE_AA)

        # Draw right hand (orange)
        if rhand:
            for a, b in HAND_CONNECTIONS:
                if a < len(rhand) and b < len(rhand):
                    pa, pb = to_px(rhand[a]), to_px(rhand[b])
                    cv2.line(img, pa, pb, (160, 100, 40), 2, cv2.LINE_AA)
                    cv2.line(img, pa, pb, (255, 165, 75), 1, cv2.LINE_AA)
            for lm in rhand:
                cv2.circle(img, to_px(lm), 3, (255, 165, 75), -1, cv2.LINE_AA)

        # Draw left hand (green)
        if lhand:
            for a, b in HAND_CONNECTIONS:
                if a < len(lhand) and b < len(lhand):
                    pa, pb = to_px(lhand[a]), to_px(lhand[b])
                    cv2.line(img, pa, pb, (80, 160, 40), 2, cv2.LINE_AA)
                    cv2.line(img, pa, pb, (168, 255, 75), 1, cv2.LINE_AA)
            for lm in lhand:
                cv2.circle(img, to_px(lm), 3, (168, 255, 75), -1, cv2.LINE_AA)

        # Label bar at top
        cv2.rectangle(img, (0, 0), (W, 28), (0, 0, 0), -1)
        cv2.putText(img, sign_label, (10, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (168, 255, 75), 1, cv2.LINE_AA)
        cv2.putText(img, 'Emirates Sign Language', (W - 185, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 130), 1, cv2.LINE_AA)

        # Progress bar
        prog = int(W * 0.9 * (fi + 1) / TOTAL)
        cv2.rectangle(img, (int(W*0.05), H-5), (int(W*0.05)+int(W*0.9), H-1), (25, 24, 42), -1)
        cv2.rectangle(img, (int(W*0.05), H-5), (int(W*0.05)+prog, H-1), (124, 58, 237), -1)

        out.write(img)

    out.release()

    # Convert to H264 MP4
    subprocess.run([
        'ffmpeg', '-y', '-i', tmp,
        '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-pix_fmt', 'yuv420p',
        str(out_path)
    ], capture_output=True)
    os.unlink(tmp)

    print(f"[SkeletonRenderer] {sign} → {out_path} ({out_path.stat().st_size//1024}KB)")
    return str(out_path)


def get_or_render(sign: str) -> str | None:
    """Return cached video path or render on demand."""
    out_path = VIDEOS_DIR / f"{sign.upper()}.mp4"
    if out_path.exists():
        return str(out_path)
    return render_skeleton_video(sign)


if __name__ == "__main__":
    # Pre-render all available signs
    import sys
    signs = [p.stem for p in MOCAP_DIR.glob("*.json")]
    print(f"Pre-rendering {len(signs)} skeleton videos...")
    for sign in sorted(signs):
        get_or_render(sign)
    print("Done.")
