#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
node render.js \
  ../../data/avatars/arab-man/arab_sheik_idle.glb \
  /root/.openclaw/workspace/arab_sheik_idle.mp4 \
  --fps 24 --w 600 --h 700 --duration 6
