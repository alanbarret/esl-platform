#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."

WORD="${1:-HOW_ARE_YOU}"
SOURCE_VIDEO="data/motion_db/${WORD}.mp4"
HOLISTIC_JSON="data/processed/mocap_holistic/${WORD}.json"
ANIM_OUT="data/avatars/arab-man/_${WORD}_anim.glb"
MERGED_OUT="data/avatars/arab-man/arab_sheik_${WORD}.glb"
VIDEO_OUT="/root/.openclaw/workspace/arab_sheik_${WORD}.mp4"
AVATAR="data/avatars/arab-man/source/ready player me arab sheik.glb"

if [ ! -f "$SOURCE_VIDEO" ]; then
  echo "Source video not found: $SOURCE_VIDEO"
  echo "Available words:"
  ls data/motion_db/ | grep -v _avatar | sed 's/.mp4$//' | column
  exit 1
fi

mkdir -p data/processed/mocap_holistic

if [ ! -f "$HOLISTIC_JSON" ]; then
  echo "=== 1/4 Extracting MediaPipe Holistic (body + hands) ==="
  python3 scripts/animate/extract_holistic.py "$SOURCE_VIDEO" "$HOLISTIC_JSON"
else
  echo "=== Using cached holistic data: $HOLISTIC_JSON ==="
fi

echo ""
echo "=== 2/4 Retargeting -> Mixamo bones ==="
python3 scripts/animate/retarget_from_mocap.py "$HOLISTIC_JSON" "$ANIM_OUT" --smooth 5 --trim-trailing

echo ""
echo "=== 3/4 Merging into Arab avatar ==="
python3 scripts/animate/merge_animation.py "$AVATAR" "$ANIM_OUT" "$MERGED_OUT"

echo ""
echo "=== 4/4 Rendering MP4 ==="
node scripts/animate/render.js "$MERGED_OUT" "$VIDEO_OUT" --fps 25 --w 600 --h 700

echo ""
echo "=== Done ==="
ls -la "$VIDEO_OUT"
