#!/usr/bin/env python3
"""
Batch-run the 3D avatar pipeline on every downloaded sign.

For each TOKEN with a source video in data/motion_db/:
  1. extract MediaPipe Holistic landmarks  (cached)
  2. retarget to the Arab sheikh GLB       (cached)
  3. merge animation into avatar GLB       (cached)
  4. render to MP4                         (cached)

Already-built outputs are skipped, so re-running the batch is cheap.

Parallelism: 2 workers by default (each render uses Chromium + ffmpeg —
running too many in parallel will starve the GPU/CPU).

Usage:
  python3 scripts/batch_render.py                  # all signs
  python3 scripts/batch_render.py --limit 50       # first 50
  python3 scripts/batch_render.py --workers 4
  python3 scripts/batch_render.py --tokens SCHOOL DOCTOR FAMILY
"""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path


BASE = Path(__file__).resolve().parent.parent
MOTION_DB = BASE / "data" / "motion_db"
GLB_DIR = BASE / "data" / "avatars" / "arab-man"
VIDEO_OUT_DIR = BASE / "data" / "avatar_videos_3d"
BUILD_SH = BASE / "scripts" / "animate" / "build.sh"


def build_one(token: str) -> tuple[str, bool, str, float]:
    """Run scripts/animate/build.sh for one token. Returns (token, ok, message, duration)."""
    t0 = time.time()
    merged_glb = GLB_DIR / f"arab_sheik_{token}.glb"
    video_out = VIDEO_OUT_DIR / f"arab_sheik_{token}.mp4"
    if (
        merged_glb.exists() and merged_glb.stat().st_size > 5_000
        and video_out.exists() and video_out.stat().st_size > 5_000
    ):
        return (token, True, "cached", time.time() - t0)
    try:
        r = subprocess.run(
            ["bash", str(BUILD_SH), token],
            cwd=BASE,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if r.returncode != 0:
            tail = (r.stderr or r.stdout or "")[-300:]
            return (token, False, f"exit {r.returncode}: {tail.strip()}", time.time() - t0)
        if not (video_out.exists() and video_out.stat().st_size > 5_000):
            return (token, False, "no output video", time.time() - t0)
        return (token, True, f"{video_out.stat().st_size // 1024} KB", time.time() - t0)
    except subprocess.TimeoutExpired:
        return (token, False, "timeout (>300s)", time.time() - t0)
    except Exception as e:
        return (token, False, str(e), time.time() - t0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tokens", nargs="+", default=None,
                    help="Only render these tokens (case-insensitive, matched to motion_db files)")
    args = ap.parse_args()

    if not BUILD_SH.exists():
        print(f"ERROR: build script missing: {BUILD_SH}", file=sys.stderr)
        return 1

    # Find available tokens
    if args.tokens:
        tokens = [t.upper() for t in args.tokens]
        tokens = [t for t in tokens if (MOTION_DB / f"{t}.mp4").exists()]
    else:
        tokens = sorted({p.stem.upper() for p in MOTION_DB.glob("*.mp4") if "_avatar" not in p.stem})
    if args.limit:
        tokens = tokens[: args.limit]
    print(f"[batch_render] {len(tokens)} tokens to process (workers={args.workers})")

    t_start = time.time()
    ok = cached = fail = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(build_one, t): t for t in tokens}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            token, success, msg, dt = fut.result()
            if success:
                if msg == "cached":
                    cached += 1
                else:
                    ok += 1
                marker = "✓"
            else:
                fail += 1
                marker = "✗"
            print(f"  {marker} [{i:>4d}/{len(tokens)}] {token:30s} {dt:5.1f}s  {msg}")

    dt = time.time() - t_start
    print()
    print(f"[batch_render] done in {dt:.1f}s · rendered={ok} cached={cached} fail={fail}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
