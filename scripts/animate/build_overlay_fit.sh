#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
WORD="${1:-HOW_ARE_YOU}"
SRC="data/motion_db/${WORD}.mp4"
HOL="data/processed/mocap_holistic/${WORD}.json"
AV="/root/.openclaw/workspace/arab_sheik_${WORD}_v2.mp4"
BONES="/root/.openclaw/workspace/arab_sheik_${WORD}_v2.bones.json"
OUT="/root/.openclaw/workspace/arab_sheik_${WORD}_overlay_fit.mp4"

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

python3 scripts/animate/overlay_landmarks_fit.py "$AV" "$SRC" "$HOL" "$BONES" "$OUT" --trim-start "$TRIM_START"
ls -la "$OUT"
