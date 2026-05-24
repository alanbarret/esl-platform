#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."

WORD="${1:-DOCTOR}"
SUFFIX="${2:-digiv2}"
SRC="data/motion_db/${WORD}.mp4"
HOLISTIC="data/processed/mocap_holistic_v2/${WORD}.json"
AV="/root/.openclaw/workspace/arab_sheik_${WORD}_${SUFFIX}.mp4"
PC="/root/.openclaw/workspace/pointcloud_${WORD}.mp4"
OUT="/root/.openclaw/workspace/triple_${WORD}_${SUFFIX}.mp4"

# Determine trim-start (same logic as the retargeter)
python3 - "$HOLISTIC" <<'PY' > /tmp/_trim_start
import json, sys
with open(sys.argv[1]) as f: d = json.load(f)
LM_LW, LM_RW = 15, 16
poses = d['frames']
start = 0
def has_signal(p):
    if not p or len(p) <= LM_RW: return False
    return p[LM_LW][3] > 0.5 or p[LM_RW][3] > 0.5
while start < len(poses) and (not poses[start].get('pose') or not has_signal(poses[start]['pose'])):
    start += 1
end = len(poses)
while end > start and (not poses[end-1].get('pose') or not has_signal(poses[end-1]['pose'])):
    end -= 1
start = max(0, start - 3)
end = min(len(poses), end + 3)
print(f"{start} {end}")
PY
read TRIM_START TRIM_END < /tmp/_trim_start
echo "Trim: $TRIM_START..$TRIM_END"

if [ ! -f "$PC" ] || [ "$3" = "force" ]; then
  echo "=== Rendering point cloud ==="
  node scripts/animate/render_pointcloud.js "$HOLISTIC" "$PC" --fps 25 --w 600 --h 700 --trim-start "$TRIM_START" --trim-end "$TRIM_END"
fi

# Get avatar duration to clip the source to match
AV_DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$AV")
SRC_FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$SRC")
SRC_FPS_NUM=$(echo "$SRC_FPS" | awk -F/ '{ printf "%.4f", $1/$2 }')
SRC_START=$(python3 -c "print(${TRIM_START} / ${SRC_FPS_NUM})")
echo "Source clip start: ${SRC_START}s, duration: ${AV_DUR}s"

echo "=== Combining 3 videos ==="
ffmpeg -y -ss "$SRC_START" -i "$SRC" -i "$PC" -i "$AV" -filter_complex \
  "[0:v]scale=600:700:force_original_aspect_ratio=decrease,pad=600:700:(ow-iw)/2:(oh-ih)/2,setsar=1[a]; \
   [1:v]scale=600:700:force_original_aspect_ratio=decrease,pad=600:700:(ow-iw)/2:(oh-ih)/2,setsar=1[b]; \
   [2:v]scale=600:700:force_original_aspect_ratio=decrease,pad=600:700:(ow-iw)/2:(oh-ih)/2,setsar=1[c]; \
   [a][b][c]hstack=inputs=3:shortest=1" \
  -c:v libx264 -crf 20 -pix_fmt yuv420p -t "$AV_DUR" -an "$OUT" 2>&1 | tail -3
ls -la "$OUT"
