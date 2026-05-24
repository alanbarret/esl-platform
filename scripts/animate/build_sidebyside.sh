#!/usr/bin/env bash
set -e
WORD="${1:-DOCTOR}"
SUFFIX="${2:-digi}"  # 'digi' or 'digiv2' etc.
SRC="/root/.openclaw/workspace/esl-platform/data/motion_db/${WORD}.mp4"
AV="/root/.openclaw/workspace/arab_sheik_${WORD}_${SUFFIX}.mp4"
OUT="/root/.openclaw/workspace/sidebyside_${WORD}_${SUFFIX}.mp4"

# Get avatar duration, match source clip
AV_DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$AV")
echo "Avatar duration: $AV_DUR s"

ffmpeg -y -i "$SRC" -i "$AV" -filter_complex \
  "[0:v]scale=600:700:force_original_aspect_ratio=decrease,pad=600:700:(ow-iw)/2:(oh-ih)/2,setsar=1[a]; \
   [1:v]scale=600:700:force_original_aspect_ratio=decrease,pad=600:700:(ow-iw)/2:(oh-ih)/2,setsar=1[b]; \
   [a][b]hstack=shortest=1" \
  -c:v libx264 -crf 20 -pix_fmt yuv420p -t "$AV_DUR" "$OUT" 2>&1 | tail -5

ls -la "$OUT"
