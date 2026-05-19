#!/bin/bash
# Watchdog: keeps demo_server.py alive
while true; do
    if ! curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo "[watchdog] Backend down, restarting..."
        fuser -k 8001/tcp 2>/dev/null
        sleep 1
        python3 /root/.openclaw/workspace/esl-platform/backend/demo_server.py &
        sleep 5
    fi
    sleep 15
done
