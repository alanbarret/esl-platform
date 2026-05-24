#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."

AVATAR="data/avatars/arab-man/source/ready player me arab sheik.glb"
OUT_DIR="data/avatars/arab-man"

echo "=== Building expressions clip ==="
python3 scripts/animate/merge_animation.py \
  "$AVATAR" \
  data/reference_animations/M_Standing_Expressions_012.glb \
  "$OUT_DIR/arab_sheik_expressions.glb"

echo ""
echo "=== Done. Files: ==="
ls -la "$OUT_DIR"/*.glb
