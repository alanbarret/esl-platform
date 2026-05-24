#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."

INPUT="/root/.openclaw/workspace/esl_realdata.avi"
ANIM_OUT="data/avatars/arab-man/_retargeted_anim.glb"
MERGED_OUT="data/avatars/arab-man/arab_sheik_retargeted.glb"
VIDEO_OUT="/root/.openclaw/workspace/arab_sheik_retargeted.mp4"
AVATAR="data/avatars/arab-man/source/ready player me arab sheik.glb"

echo "=== 1/3 MediaPipe retargeting (5s test) ==="
python3 scripts/animate/retarget_mediapipe.py "$INPUT" "$ANIM_OUT" --max-seconds 5 --smooth 5

echo ""
echo "=== 2/3 Merging into Arab avatar ==="
python3 scripts/animate/merge_animation.py "$AVATAR" "$ANIM_OUT" "$MERGED_OUT"

echo ""
echo "=== 3/3 Rendering MP4 ==="
node scripts/animate/render.js "$MERGED_OUT" "$VIDEO_OUT" --fps 30 --w 600 --h 700

echo ""
echo "=== Done ==="
ls -la "$VIDEO_OUT"
