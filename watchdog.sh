#!/bin/bash
# ESL Platform Watchdog — auto-restarts backend if it crashes
# Usage: bash watchdog.sh &

BACKEND_SCRIPT="/root/.openclaw/workspace/esl-platform/backend/lean_server.py"
LOG="/tmp/esl_backend.log"
CHECK_INTERVAL=10
BACKEND_PID=""

start_backend() {
    echo "[watchdog] $(date '+%H:%M:%S') Starting backend..."
    python3 "$BACKEND_SCRIPT" >> "$LOG" 2>&1 &
    BACKEND_PID=$!
    echo "[watchdog] Backend PID: $BACKEND_PID"
}

echo "[watchdog] Started — checking every ${CHECK_INTERVAL}s"
start_backend

while true; do
    sleep $CHECK_INTERVAL

    # Check if process is alive
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "[watchdog] $(date '+%H:%M:%S') Backend died (PID $BACKEND_PID), restarting..."
        # Kill anything still on port 8001
        fuser -k 8001/tcp 2>/dev/null
        sleep 2
        start_backend
        continue
    fi

    # Also check HTTP health
    if ! curl -sf http://localhost:8001/health > /dev/null 2>&1; then
        echo "[watchdog] $(date '+%H:%M:%S') Health check failed, restarting..."
        kill $BACKEND_PID 2>/dev/null
        fuser -k 8001/tcp 2>/dev/null
        sleep 2
        start_backend
    fi
done
