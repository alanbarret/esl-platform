#!/usr/bin/env python3
"""
Scrape sign-language videos from the UAE Sign Language manifest and produce:
  1. Source MP4s          (data/motion_db/<TOKEN>.mp4)
  2. Per-frame mocap JSON (data/processed/mocap/<TOKEN>.json) — body + 2 hands
  3. Skeleton video       (data/skeleton_videos/<TOKEN>.mp4) — wireframe overlay
     trimmed to frames where at least one hand is visible.

The mocap JSON + skeleton videos are what the WordPress plugin and the
platform's skeleton endpoints serve. The MP4 source is what the 3D avatar
pipeline (`scripts/batch_render.py`) consumes.

Token normalization:
  "Archery"      -> ARCHERY
  "Rowing Boat"  -> ROWING_BOAT

Already-completed steps are skipped, so re-running is cheap.

Usage:
  python3 scripts/scrape.py
  python3 scripts/scrape.py --limit 50
  python3 scripts/scrape.py --filter sport
  python3 scripts/scrape.py --workers 4
  python3 scripts/scrape.py --skip-skeleton    # only download + mocap
  python3 scripts/scrape.py --skip-mocap       # only download
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path


BASE = Path(__file__).resolve().parent.parent
MANIFEST = BASE / "data" / "raw" / "uae_signs_full.json"
MOTION_DB = BASE / "data" / "motion_db"
MOCAP_DIR = BASE / "data" / "processed" / "mocap"
SKELETON_DIR = BASE / "data" / "skeleton_videos"


def normalize_token(name: str) -> str:
    return (name or "").strip().upper().replace(" ", "_").replace("/", "_").replace("-", "_")


def download(entry: dict, force: bool = False) -> tuple[Path | None, str]:
    name = entry.get("english") or entry.get("name") or ""
    url = entry.get("video_url")
    token = normalize_token(name)
    if not token or not url:
        return (None, "missing name/url")
    dst = MOTION_DB / f"{token}.mp4"
    if not force and dst.exists() and dst.stat().st_size > 10_000:
        return (dst, "cached")
    try:
        r = subprocess.run(
            ["curl", "-sL", "--max-time", "30", "-o", str(dst), url],
            capture_output=True, timeout=40,
        )
        if r.returncode != 0:
            return (None, f"curl exit {r.returncode}")
        if not dst.exists() or dst.stat().st_size < 10_000:
            return (None, f"too small ({dst.stat().st_size if dst.exists() else 0} bytes)")
        return (dst, f"{dst.stat().st_size // 1024} KB")
    except subprocess.TimeoutExpired:
        return (None, "timeout")
    except Exception as e:
        return (None, str(e))


def extract_mocap_and_skeleton(token: str, video_path: Path,
                                want_mocap: bool, want_skeleton: bool) -> str:
    """Use MediaPipe Holistic to extract body+hand landmarks from the source video,
    write mocap JSON and (optionally) render a skeleton overlay video.
    """
    if not (want_mocap or want_skeleton):
        return "skipped"

    mocap_path = MOCAP_DIR / f"{token}.json"
    skel_path = SKELETON_DIR / f"{token}.mp4"

    have_mocap = mocap_path.exists() and mocap_path.stat().st_size > 200
    have_skel = skel_path.exists() and skel_path.stat().st_size > 5000

    if (not want_mocap or have_mocap) and (not want_skeleton or have_skel):
        return "cached"

    # Lazy import: heavyweight
    import cv2
    import numpy as np
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return "open-fail"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 3:
        cap.release()
        return "too-short"

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames_data = []
    has_hand = []
    src_frames: list[np.ndarray] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holistic.process(rgb)
        fd: dict[str, list] = {}
        # pose_landmarks: image-normalized (0..1) — used by skeleton wireframe + 2D overlays
        if res.pose_landmarks:
            fd["pose"] = [[lm.x, lm.y, lm.z, lm.visibility] for lm in res.pose_landmarks.landmark]
        # pose_world_landmarks: metric meters — used by the 3D retargeter
        if res.pose_world_landmarks:
            fd["pose_world"] = [[lm.x, lm.y, lm.z, lm.visibility] for lm in res.pose_world_landmarks.landmark]
        if res.left_hand_landmarks:
            fd["lhand"] = [[lm.x, lm.y, lm.z] for lm in res.left_hand_landmarks.landmark]
        if res.right_hand_landmarks:
            fd["rhand"] = [[lm.x, lm.y, lm.z] for lm in res.right_hand_landmarks.landmark]
        frames_data.append(fd)
        has_hand.append(bool(fd.get("lhand") or fd.get("rhand")))
        if want_skeleton:
            src_frames.append(frame)
    cap.release()
    holistic.close()

    # Write mocap JSON
    if want_mocap and not have_mocap:
        MOCAP_DIR.mkdir(parents=True, exist_ok=True)
        mocap_path.write_text(json.dumps({"fps": fps, "frames": frames_data}))

    # Render skeleton video
    if want_skeleton and not have_skel:
        _render_skeleton(token, frames_data, has_hand, src_frames, fps)

    return "ok"


def _render_skeleton(token: str, frames_data: list[dict], has_hand: list[bool],
                     src_frames, fps: float) -> None:
    """Render a skeleton-wireframe MP4 trimmed to frames where hands are visible."""
    import cv2
    import numpy as np

    POSE_CONN = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (24, 26), (26, 28),
    ]
    HAND_CONN = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]
    W, H = 640, 360
    BUFFER = 5
    n = len(has_hand)
    expanded = list(has_hand)
    for i, v in enumerate(has_hand):
        if v:
            for j in range(max(0, i - BUFFER), min(n, i + BUFFER + 1)):
                expanded[j] = True
    keep = [i for i, v in enumerate(expanded) if v] or list(range(n))

    SKELETON_DIR.mkdir(parents=True, exist_ok=True)
    tmp_avi = SKELETON_DIR / f"{token}.tmp.avi"
    writer = cv2.VideoWriter(str(tmp_avi), cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))

    for fi in keep:
        fd = frames_data[fi]
        img = np.zeros((H, W, 3), dtype=np.uint8)

        def draw_landmarks(lms, conns, color, scale_x, scale_y, off_x=0, off_y=0):
            if not lms: return
            pts = [(int(lm[0] * scale_x + off_x), int(lm[1] * scale_y + off_y)) for lm in lms]
            for a, b in conns:
                if a < len(pts) and b < len(pts):
                    cv2.line(img, pts[a], pts[b], color, 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(img, p, 3, color, -1)

        # Use pose landmarks (image-normalized 0..1 — same coord system as hand landmarks).
        # Older files stored pose_world (-0.5..0.5 meters); detect and handle both.
        pose = fd.get("pose")
        pose_is_world = pose and (pose[0][0] < 0 or pose[0][1] < 0)

        if pose_is_world:
            # World coords: scale + centre to fit the frame.
            pose_scale = H * 0.85
            pose_off_x = W // 2
            pose_off_y = int(H * 0.65)
            def pose_to_px(lm):
                return int(lm[0] * pose_scale + pose_off_x), int(lm[1] * pose_scale + pose_off_y)
        else:
            # Image-normalized 0..1: map directly to pixels.
            def pose_to_px(lm):
                return int(lm[0] * W), int(lm[1] * H)

        if pose:
            # Body skeleton
            pts = [pose_to_px(lm) for lm in pose]
            for a, b in POSE_CONN:
                if a < len(pts) and b < len(pts):
                    cv2.line(img, pts[a], pts[b], (255, 80, 220), 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(img, p, 3, (255, 80, 220), -1)

        def draw_hand_attached(hand_lm, pose_wrist_idx, pose_elbow_idx, color):
            if not hand_lm: return
            if not pose or len(pose) <= pose_wrist_idx: return
            wrist_px = pose_to_px(pose[pose_wrist_idx])
            elbow_px = pose_to_px(pose[pose_elbow_idx])
            forearm_len = ((wrist_px[0]-elbow_px[0])**2 + (wrist_px[1]-elbow_px[1])**2) ** 0.5
            if forearm_len < 5: return
            # Compute hand's own extent (lm 0 = wrist, lm 12 = middle fingertip)
            ref_size = ((hand_lm[12][0]-hand_lm[0][0])**2 + (hand_lm[12][1]-hand_lm[0][1])**2) ** 0.5
            if ref_size < 0.01: return
            # Scale: hand should be ~70% of forearm length in pixels
            scale = (forearm_len * 0.7) / ref_size
            def hand_to_px(lm):
                dx = (lm[0] - hand_lm[0][0]) * scale
                dy = (lm[1] - hand_lm[0][1]) * scale
                return int(wrist_px[0] + dx), int(wrist_px[1] + dy)
            pts = [hand_to_px(lm) for lm in hand_lm]
            for a, b in HAND_CONN:
                if a < len(pts) and b < len(pts):
                    cv2.line(img, pts[a], pts[b], color, 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(img, p, 3, color, -1)

        lh = fd.get("lhand")
        rh = fd.get("rhand")
        draw_hand_attached(lh, 15, 13, (80, 255, 80))   # MediaPipe pose: 15=L wrist, 13=L elbow
        draw_hand_attached(rh, 16, 14, (80, 200, 255))  # 16=R wrist, 14=R elbow

        writer.write(img)
    writer.release()

    # Re-encode to H.264. Keep 640x360 (no crop) so we can debug.
    out_mp4 = SKELETON_DIR / f"{token}.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(tmp_avi),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
         str(out_mp4)],
        capture_output=True,
    )
    tmp_avi.unlink(missing_ok=True)


def process_one(entry: dict, want_mocap: bool, want_skeleton: bool,
                force: bool) -> tuple[str, str]:
    token = normalize_token(entry.get("english") or entry.get("name") or "")
    if not token:
        return ("?", "missing name")
    path, msg = download(entry, force=force)
    if path is None:
        return (token, f"download: {msg}")
    if want_mocap or want_skeleton:
        msg2 = extract_mocap_and_skeleton(token, path, want_mocap, want_skeleton)
        return (token, f"download:{msg} mocap/skel:{msg2}")
    return (token, f"download:{msg}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--filter", default=None,
                    help="Only entries whose english/category contains this substring")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel workers for the download+extract step (default 4)")
    ap.add_argument("--force", action="store_true", help="Re-download even if cached")
    ap.add_argument("--skip-mocap", action="store_true",
                    help="Only download MP4s; skip MediaPipe mocap extraction")
    ap.add_argument("--skip-skeleton", action="store_true",
                    help="Skip skeleton wireframe video rendering")
    args = ap.parse_args()

    if not MANIFEST.exists():
        print(f"ERROR: manifest not found at {MANIFEST}", file=sys.stderr)
        return 1
    for d in (MOTION_DB, MOCAP_DIR, SKELETON_DIR):
        d.mkdir(parents=True, exist_ok=True)

    entries = json.loads(MANIFEST.read_text())
    if args.filter:
        f = args.filter.lower()
        entries = [
            e for e in entries
            if f in (e.get("english", "").lower()) or f in (e.get("category", "").lower())
        ]
    if args.limit:
        entries = entries[: args.limit]

    want_mocap = not args.skip_mocap
    want_skeleton = not args.skip_skeleton
    print(f"[scrape] {len(entries)} entries · workers={args.workers} · mocap={want_mocap} skeleton={want_skeleton}")

    t0 = time.time()
    ok = fail = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, e, want_mocap, want_skeleton, args.force): e for e in entries}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            token, msg = fut.result()
            if "download:" in msg and ("KB" in msg or "cached" in msg):
                ok += 1
                marker = "✓"
            else:
                fail += 1
                marker = "✗"
            print(f"  {marker} [{i:>4d}/{len(entries)}] {token:30s} {msg}")

    dt = time.time() - t0
    print()
    print(f"[scrape] done in {dt:.1f}s · ok={ok} fail={fail}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
