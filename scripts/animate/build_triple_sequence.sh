#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."

# Args: a list of "tokens" — each token is either a single word or a comma-separated chain
# of words (which gets concatenated as one unit). Per token we produce three clips:
#   - original  (data/motion_db/<TOKEN>.mp4 if exists, else a black placeholder)
#   - skeleton  (data/skeleton_videos/<TOKEN>.mp4 if exists)
#   - avatar 3D (data/avatar_videos_3d/arab_sheik_<TOKEN>.mp4, rendered if needed)
# Then concat all per-row clips and combine in three side-by-side panels.

TOKENS_INPUT=("$@")
if [ ${#TOKENS_INPUT[@]} -eq 0 ]; then
  echo "Usage: $0 TOKEN[,TOKEN,...] [TOKEN[,TOKEN,...] ...]"
  echo "Example: $0 SCHOOL CLOSE AEEN,ALIF,AEEN MEEM,TAA,RAA"
  exit 1
fi

OUT_NAME="triple_$(echo "${TOKENS_INPUT[@]}" | tr ' ,' '__' | tr -d 'أإآ')"
WORK="/tmp/triple_$$"
mkdir -p "$WORK"

# Helpers
panel_w=600; panel_h=700
fps=25

render_3d() {
  # $1 = token. Output: /root/.openclaw/workspace/esl-platform/data/avatar_videos_3d/arab_sheik_$1.mp4
  local tok="$1"
  local src="data/motion_db/${tok}.mp4"
  local out="data/avatar_videos_3d/arab_sheik_${tok}.mp4"
  if [ -f "$out" ] && [ $(stat -c%s "$out") -gt 5000 ]; then echo "  (cached) $out"; return; fi
  if [ ! -f "$src" ]; then echo "  NO 3D source for $tok"; return; fi
  bash scripts/animate/build_digihuman_v2.sh "$tok" >/dev/null 2>&1
  # Copy from /root/.openclaw/workspace location
  cp "/root/.openclaw/workspace/arab_sheik_${tok}_digiv2.mp4" "$out" 2>/dev/null || true
}

declare -a row_orig row_skel row_av

for tok_group in "${TOKENS_INPUT[@]}"; do
  IFS=',' read -ra parts <<< "$tok_group"
  group_orig=""; group_skel=""; group_av=""
  for tok in "${parts[@]}"; do
    tok="$(echo "$tok" | tr -d '[:space:]')"
    src="data/motion_db/${tok}.mp4"
    skel="data/skeleton_videos/${tok}.mp4"
    echo ""
    echo "=== Token: $tok ==="
    # 3D render
    render_3d "$tok"
    av="data/avatar_videos_3d/arab_sheik_${tok}.mp4"
    # Track which clips we have
    if [ -f "$src" ]; then group_orig="$group_orig $src"; fi
    if [ -f "$skel" ]; then group_skel="$group_skel $skel"; fi
    if [ -f "$av" ] && [ $(stat -c%s "$av") -gt 5000 ]; then group_av="$group_av $av"; fi
  done
  row_orig+=("$group_orig")
  row_skel+=("$group_skel")
  row_av+=("$group_av")
done

# Concat each row into a single clip
make_row_clip() {
  # $1 = label (orig/skel/av), $2... = video files
  local label="$1"; shift
  local files=("$@")
  local out="$WORK/${label}_concat.mp4"
  if [ ${#files[@]} -eq 0 ]; then
    # Black placeholder ${panel_w}x${panel_h}
    ffmpeg -y -f lavfi -i "color=c=black:s=${panel_w}x${panel_h}:d=2:r=${fps}" \
      -c:v libx264 -crf 23 -pix_fmt yuv420p "$out" 2>/dev/null
    echo "$out"
    return
  fi
  # Normalize each clip and build list
  local listfile="$WORK/${label}.txt"; > "$listfile"
  local i=0
  for f in "${files[@]}"; do
    norm="$WORK/${label}_${i}.mp4"
    ffmpeg -y -i "$f" \
      -vf "scale=${panel_w}:${panel_h}:force_original_aspect_ratio=decrease,pad=${panel_w}:${panel_h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=${fps}" \
      -c:v libx264 -crf 23 -pix_fmt yuv420p -an "$norm" 2>/dev/null
    echo "file '$norm'" >> "$listfile"
    i=$((i+1))
  done
  ffmpeg -y -f concat -safe 0 -i "$listfile" -c copy "$out" 2>/dev/null
  echo "$out"
}

# Flatten all groups into one big sequential clip per row
declare -a all_orig all_skel all_av
for g in "${row_orig[@]}"; do for f in $g; do all_orig+=("$f"); done; done
for g in "${row_skel[@]}"; do for f in $g; do all_skel+=("$f"); done; done
for g in "${row_av[@]}";   do for f in $g; do all_av+=("$f"); done; done

echo ""
echo "=== Concatenating rows ==="
orig_clip=$(make_row_clip orig "${all_orig[@]}")
skel_clip=$(make_row_clip skel "${all_skel[@]}")
av_clip=$(make_row_clip av "${all_av[@]}")
echo "orig: $orig_clip ($(ls -l "$orig_clip" | awk '{print $5}') bytes)"
echo "skel: $skel_clip ($(ls -l "$skel_clip" | awk '{print $5}') bytes)"
echo "av:   $av_clip ($(ls -l "$av_clip" | awk '{print $5}') bytes)"

# Determine the longest clip's duration so each row pads to that
get_dur() { ffprobe -v error -show_entries format=duration -of csv=p=0 "$1" 2>/dev/null; }
MAX=$(python3 -c "print(max([float('$(get_dur "$orig_clip")' or 0), float('$(get_dur "$skel_clip")' or 0), float('$(get_dur "$av_clip")' or 0)]))")
echo "Max duration: $MAX s"

echo ""
echo "=== Final hstack ==="
OUT="/root/.openclaw/workspace/${OUT_NAME}.mp4"
ffmpeg -y -i "$orig_clip" -i "$skel_clip" -i "$av_clip" -filter_complex \
  "[0:v]tpad=stop_duration=${MAX},setpts=PTS-STARTPTS,scale=${panel_w}:${panel_h},drawtext=text='ORIGINAL':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5[a]; \
   [1:v]tpad=stop_duration=${MAX},setpts=PTS-STARTPTS,scale=${panel_w}:${panel_h},drawtext=text='SKELETON':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5[b]; \
   [2:v]tpad=stop_duration=${MAX},setpts=PTS-STARTPTS,scale=${panel_w}:${panel_h},drawtext=text='3D AVATAR':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5[c]; \
   [a][b][c]hstack=inputs=3:shortest=0" \
  -c:v libx264 -crf 20 -pix_fmt yuv420p -t "$MAX" "$OUT" 2>&1 | tail -3

ls -la "$OUT"
rm -rf "$WORK"
