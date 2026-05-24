#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
node render.js \
  ../../data/avatars/arab-man/arab_sheik_expressions.glb \
  /root/.openclaw/workspace/arab_sheik_expressions.mp4 \
  --fps 24 --w 600 --h 700
