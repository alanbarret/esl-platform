#!/usr/bin/env bash
# Build the 3D avatar pipeline for a single token:
#   source video → MediaPipe Holistic → DigiHuman retarget → merged GLB → MP4
#
# Output paths match what backend/server.py expects to find:
#   data/processed/mocap_holistic_v2/{TOKEN}.json
#   data/avatars/arab-man/_{TOKEN}_anim.glb
#   data/avatars/arab-man/arab_sheik_{TOKEN}.glb
#   data/avatar_videos_3d/arab_sheik_{TOKEN}.mp4
#
# Usage: build.sh <TOKEN>
set -e
cd "$(dirname "$0")/../.."

WORD="${1:?usage: build.sh <TOKEN>}"
SOURCE_VIDEO="data/motion_db/${WORD}.mp4"
HOLISTIC_JSON="data/processed/mocap_holistic_v2/${WORD}.json"
ANIM_OUT="data/avatars/arab-man/_${WORD}_anim.glb"
MERGED_OUT="data/avatars/arab-man/arab_sheik_${WORD}.glb"
VIDEO_OUT="data/avatar_videos_3d/arab_sheik_${WORD}.mp4"
AVATAR="data/avatars/arab-man/source/ESL_Avatar.glb"

if [ ! -f "$SOURCE_VIDEO" ]; then
  echo "ERROR: source video missing: $SOURCE_VIDEO"
  echo "Run scripts/scrape.py first to download the manifest videos."
  exit 1
fi

mkdir -p "$(dirname "$HOLISTIC_JSON")" "$(dirname "$VIDEO_OUT")"

if [ ! -f "$HOLISTIC_JSON" ]; then
  echo "[1/4] Extracting MediaPipe Holistic (Tasks API: pose + 2 hands in metric 3D)..."
  python3 scripts/animate/extract_v2.py "$SOURCE_VIDEO" "$HOLISTIC_JSON"
fi

echo "[2/4] Retargeting (DigiHuman LookRotation)..."
python3 scripts/animate/retarget_digihuman.py "$AVATAR" "$HOLISTIC_JSON" "$ANIM_OUT" \
  --smooth 5 --trim-trailing

echo "[3/4] Merging animation into avatar GLB..."
python3 scripts/animate/merge_animation.py "$AVATAR" "$ANIM_OUT" "$MERGED_OUT"

echo "[4/4] Rendering MP4 (headless Chromium + Three.js + ffmpeg)..."
node scripts/animate/render.js "$MERGED_OUT" "$VIDEO_OUT" --fps 25 --w 600 --h 700

echo ""
echo "✅ Done:"
ls -la "$VIDEO_OUT" "$MERGED_OUT"
