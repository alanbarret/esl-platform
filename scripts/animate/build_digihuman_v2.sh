#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
WORD="${1:-DOCTOR}"
SOURCE_VIDEO="data/motion_db/${WORD}.mp4"
HOLISTIC_JSON="data/processed/mocap_holistic_v2/${WORD}.json"
ANIM_OUT="data/avatars/arab-man/_${WORD}_digiv2_anim.glb"
MERGED_OUT="data/avatars/arab-man/arab_sheik_${WORD}_digiv2.glb"
VIDEO_OUT="/root/.openclaw/workspace/arab_sheik_${WORD}_digiv2.mp4"
AVATAR="data/avatars/arab-man/source/ready player me arab sheik.glb"

mkdir -p data/processed/mocap_holistic_v2
if [ ! -f "$HOLISTIC_JSON" ]; then
  echo "=== Extracting with new MediaPipe Tasks API (3D metric world landmarks) ==="
  python3 scripts/animate/extract_v2.py "$SOURCE_VIDEO" "$HOLISTIC_JSON"
fi

echo "=== Retargeting (DigiHuman algorithm) ==="
python3 scripts/animate/retarget_digihuman.py "$AVATAR" "$HOLISTIC_JSON" "$ANIM_OUT" --smooth 5 --trim-trailing

echo "=== Merging into avatar ==="
python3 scripts/animate/merge_animation.py "$AVATAR" "$ANIM_OUT" "$MERGED_OUT"

echo "=== Rendering ==="
node scripts/animate/render.js "$MERGED_OUT" "$VIDEO_OUT" --fps 25 --w 600 --h 700

ls -la "$VIDEO_OUT"
