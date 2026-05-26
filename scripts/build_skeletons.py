#!/usr/bin/env python3
"""Regenerate mocap JSON + skeleton wireframe MP4s from already-downloaded source videos.

Useful when you have data/motion_db/<TOKEN>.mp4 but the mocap/skeleton caches
were deleted. Doesn't re-download.

Usage:
  python3 scripts/build_skeletons.py
  python3 scripts/build_skeletons.py --workers 4
"""
from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
from pathlib import Path


BASE = Path(__file__).resolve().parent.parent
MOTION_DB = BASE / "data" / "motion_db"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tokens", nargs="+", default=None)
    args = ap.parse_args()

    # Import the extraction function from scrape.py
    sys.path.insert(0, str(BASE / "scripts"))
    from scrape import extract_mocap_and_skeleton  # type: ignore

    if args.tokens:
        tokens = [t.upper() for t in args.tokens]
        tokens = [t for t in tokens if (MOTION_DB / f"{t}.mp4").exists()]
    else:
        tokens = sorted({p.stem.upper() for p in MOTION_DB.glob("*.mp4") if "_avatar" not in p.stem})

    print(f"[build_skeletons] {len(tokens)} tokens · workers={args.workers}")

    t0 = time.time()
    ok = fail = 0

    def worker(token: str) -> tuple[str, str]:
        try:
            msg = extract_mocap_and_skeleton(token, MOTION_DB / f"{token}.mp4",
                                             want_mocap=True, want_skeleton=True)
            return (token, msg)
        except Exception as e:
            return (token, f"error: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(worker, t): t for t in tokens}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            token, msg = fut.result()
            marker = "✓" if msg in ("ok", "cached") else "✗"
            if marker == "✓":
                ok += 1
            else:
                fail += 1
            print(f"  {marker} [{i:>4d}/{len(tokens)}] {token:30s} {msg}")

    dt = time.time() - t0
    print(f"\n[build_skeletons] done in {dt:.1f}s · ok={ok} fail={fail}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
