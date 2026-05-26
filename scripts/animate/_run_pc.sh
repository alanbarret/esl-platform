#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
node render_pointcloud.js "$@"
