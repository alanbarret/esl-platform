#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."

WORD="${1:-HOW_ARE_YOU}"
IDLE_ANIM="data/reference_animations/F_Standing_Idle_001.glb"
ESL_ANIM="data/avatars/arab-man/_${WORD}_anim.glb"
LAYERED_ANIM="data/avatars/arab-man/_${WORD}_layered.glb"
MERGED_OUT="data/avatars/arab-man/arab_sheik_${WORD}_layered.glb"
VIDEO_OUT="/root/.openclaw/workspace/arab_sheik_${WORD}_layered.mp4"
AVATAR="data/avatars/arab-man/source/ready player me arab sheik.glb"

if [ ! -f "$ESL_ANIM" ]; then
  echo "ESL animation not found: $ESL_ANIM"
  echo "Run build_word.sh $WORD first."
  exit 1
fi

echo "=== 1/3 Layering idle + ESL ==="
python3 scripts/animate/layer_animations.py "$IDLE_ANIM" "$ESL_ANIM" "$LAYERED_ANIM"

echo ""
echo "=== 2/3 Merging into Arab avatar ==="
python3 scripts/animate/merge_animation.py "$AVATAR" "$LAYERED_ANIM" "$MERGED_OUT"

echo ""
echo "=== 3/3 Rendering MP4 ==="
node scripts/animate/render.js "$MERGED_OUT" "$VIDEO_OUT" --fps 25 --w 600 --h 700

echo ""
echo "=== Done ==="
ls -la "$VIDEO_OUT"
