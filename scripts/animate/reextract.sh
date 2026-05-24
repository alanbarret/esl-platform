#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
WORD="${1:-HOW_ARE_YOU}"
rm -f "data/processed/mocap_holistic/${WORD}.json"
python3 scripts/animate/extract_holistic.py "data/motion_db/${WORD}.mp4" "data/processed/mocap_holistic/${WORD}.json"
