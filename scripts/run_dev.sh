#!/bin/bash
# Development runner with auto-restart and debug output

source venv/bin/activate

# Enable debug logging
export WALLIE_LOG_LEVEL=DEBUG
export PYTHONUNBUFFERED=1

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

clear
echo -e "${GREEN}🎙️  Wallie Development Mode${NC}"
echo -e "${YELLOW}Ctrl+C to stop, auto-restarts on crash${NC}"
echo ""

# Run with auto-restart
while true; do
    python wallie_voice_bot.py
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Wallie stopped normally"
        break
    else
        echo -e "${YELLOW}Wallie crashed (exit code: $EXIT_CODE), restarting in 3s...${NC}"
        sleep 3
    fi
done
