#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
WORD="${1:-HOW_ARE_YOU}"
SRC="data/motion_db/${WORD}.mp4"
HOL="data/processed/mocap_holistic/${WORD}.json"
AV="/root/.openclaw/workspace/arab_sheik_${WORD}_v2.mp4"
OUT="/root/.openclaw/workspace/arab_sheik_${WORD}_overlay.mp4"

# Need to figure out trim_start = how many frames the retargeter skipped from the start.
# It's the number of leading frames where wrist visibility was below threshold.
# Mirror that logic here:
python3 - "$HOL" <<'PY' > /tmp/_trim_start
import json, sys
with open(sys.argv[1]) as f: d = json.load(f)
LM_LW, LM_RW = 15, 16
poses = d['frames']
start = 0
def has_signal(p):
    return p[LM_LW][3] > 0.5 or p[LM_RW][3] > 0.5
while start < len(poses) and (not poses[start].get('pose') or not has_signal(poses[start]['pose'])):
    start += 1
start = max(0, start - 3)
print(start)
PY
TRIM_START=$(cat /tmp/_trim_start)
echo "Trim start: $TRIM_START"

BONES_JSON="/root/.openclaw/workspace/arab_sheik_${WORD}_v2.bones.json"
python3 scripts/animate/overlay_landmarks.py "$AV" "$SRC" "$HOL" "$OUT" --trim-start "$TRIM_START" --bones-json "$BONES_JSON"
ls -la "$OUT"
