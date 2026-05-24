#!/usr/bin/env python3
"""
Scrape sign-language videos from the UAE Sign Language manifest.

Reads:   data/raw/uae_signs_full.json   (list of {english, video_url, ...})
Writes:  data/motion_db/<TOKEN>.mp4     (one MP4 per sign)

Token normalization:
  "Archery"      -> ARCHERY
  "Rowing Boat"  -> ROWING_BOAT
  "Abu Dhabi"    -> ABU_DHABI

Skips entries that:
  - have no video_url
  - are already downloaded (file exists and > 10 KB)
  - return a non-2xx HTTP status

Parallelism: 4 workers by default (override with --workers).

Usage:
  python3 scripts/scrape.py                    # download all signs
  python3 scripts/scrape.py --limit 50         # first 50 only
  python3 scripts/scrape.py --filter sport     # only entries matching 'sport'
  python3 scripts/scrape.py --workers 8        # bump parallelism
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


def normalize_token(name: str) -> str:
    """'Rowing Boat' -> 'ROWING_BOAT'."""
    return (name or "").strip().upper().replace(" ", "_").replace("/", "_").replace("-", "_")


def download_one(entry: dict, force: bool = False) -> tuple[str, bool, str]:
    """Returns (token, ok, message)."""
    name = entry.get("english") or entry.get("name") or ""
    url = entry.get("video_url")
    token = normalize_token(name)
    if not token or not url:
        return (token, False, "missing name/url")
    dst = MOTION_DB / f"{token}.mp4"
    if not force and dst.exists() and dst.stat().st_size > 10_000:
        return (token, True, "cached")
    try:
        r = subprocess.run(
            ["curl", "-sL", "--max-time", "30", "-o", str(dst), url],
            capture_output=True, timeout=40,
        )
        if r.returncode != 0:
            return (token, False, f"curl exit {r.returncode}")
        if not dst.exists() or dst.stat().st_size < 10_000:
            return (token, False, f"too small ({dst.stat().st_size if dst.exists() else 0} bytes)")
        return (token, True, f"{dst.stat().st_size // 1024} KB")
    except subprocess.TimeoutExpired:
        return (token, False, "timeout")
    except Exception as e:
        return (token, False, str(e))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Limit to first N entries")
    ap.add_argument("--filter", default=None, help="Only entries whose english/category contains this substring (case-insensitive)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = ap.parse_args()

    if not MANIFEST.exists():
        print(f"ERROR: manifest not found at {MANIFEST}", file=sys.stderr)
        return 1
    MOTION_DB.mkdir(parents=True, exist_ok=True)

    entries = json.loads(MANIFEST.read_text())
    if args.filter:
        f = args.filter.lower()
        entries = [
            e for e in entries
            if f in (e.get("english", "").lower())
            or f in (e.get("category", "").lower())
        ]
    if args.limit:
        entries = entries[: args.limit]
    print(f"[scrape] {len(entries)} entries to process (workers={args.workers})")

    t0 = time.time()
    ok = cached = fail = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(download_one, e, args.force): e for e in entries}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            token, success, msg = fut.result()
            if success:
                if msg == "cached":
                    cached += 1
                else:
                    ok += 1
                marker = "✓"
            else:
                fail += 1
                marker = "✗"
            print(f"  {marker} [{i:>4d}/{len(entries)}] {token:30s} {msg}")

    dt = time.time() - t0
    print()
    print(f"[scrape] done in {dt:.1f}s · ok={ok} cached={cached} fail={fail}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
