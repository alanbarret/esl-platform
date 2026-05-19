#!/bin/bash
# Resume scraper only when enough RAM is free
SCRAPER_PID=$(ps aux | grep scrape_and_clip | grep -v grep | awk '{print $2}' | head -1)
if [ -z "$SCRAPER_PID" ]; then
    echo "Scraper not running"
    exit 1
fi

echo "Scraper PID: $SCRAPER_PID"
while true; do
    FREE=$(free -m | awk '/^Mem:/{print $7}')
    BACKEND=$(curl -s http://localhost:8001/health 2>/dev/null | grep -c '"ok"')
    
    if [ "$FREE" -gt 600 ]; then
        kill -CONT $SCRAPER_PID 2>/dev/null
        sleep 20
        kill -STOP $SCRAPER_PID 2>/dev/null
        sleep 5
    else
        kill -STOP $SCRAPER_PID 2>/dev/null
        echo "RAM low ($FREE MB), waiting..."
        sleep 30
    fi
done
