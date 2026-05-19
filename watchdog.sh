#!/bin/bash
# Tight watchdog: restarts backend within 5 seconds of death
SCRIPT="/root/.openclaw/workspace/esl-platform/backend/lean_server.py"
LOG="/tmp/esl_backend.log"

echo "[watchdog] Started at $(date)"

while true; do
    # Kill anything on 8001 first
    fuser -k 8001/tcp 2>/dev/null
    sleep 1
    
    echo "[watchdog] $(date '+%H:%M:%S') Starting backend..."
    python3 "$SCRIPT" >> "$LOG" 2>&1 &
    PID=$!
    echo "[watchdog] PID=$PID"
    
    # Wait for process to die
    wait $PID
    CODE=$?
    echo "[watchdog] $(date '+%H:%M:%S') Backend died (exit=$CODE), restarting in 3s..."
    sleep 3
done
