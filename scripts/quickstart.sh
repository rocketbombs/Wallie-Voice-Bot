#!/bin/bash
# Wallie Voice Bot - Quick Start Script
# One-command setup and run

set -e

YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🎙️  Wallie Voice Bot - Quick Start${NC}"
echo ""

# Check if already installed
if [ -d "venv" ] && [ -f "wallie_voice_bot.py" ]; then
    echo "Found existing installation"
    source venv/bin/activate
else
    echo "Running first-time setup..."
    
    # Check prerequisites
    if ! command -v python3.12 &> /dev/null; then
        echo -e "${RED}Error: Python 3.12+ required${NC}"
        echo "Install with: sudo apt install python3.12 python3.12-venv"
        exit 1
    fi
    
    # Quick system deps (non-interactive)
    echo "Installing system dependencies..."
    
    # Fix common apt issues on WSL
    if grep -qi microsoft /proc/version; then
        echo "Detected WSL, fixing apt..."
        sudo apt-get update 2>&1 | grep -v "apt_pkg" || true
    else
        sudo apt-get update -qq
    fi
    
    sudo apt-get install -y -qq \
        python3.12-dev \
        portaudio19-dev \
        build-essential \
        espeak \
        ffmpeg 2>/dev/null || {
            echo -e "${YELLOW}Some system packages failed to install${NC}"
            echo "Continuing anyway..."
        }
    
    # Run full install
    chmod +x scripts/install_deps.sh
    ./scripts/install_deps.sh
fi

# Activate environment
source venv/bin/activate

# Quick audio test
echo ""
echo -e "${YELLOW}Testing audio devices...${NC}"
python -c "import sounddevice as sd; print(f'Input: {sd.query_devices(kind=\"input\")[\"name\"]}'); print(f'Output: {sd.query_devices(kind=\"output\")[\"name\"]}')"

# Check GPU
echo ""
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo -e "${GREEN}✓ GPU detected${NC}"
else
    echo -e "${YELLOW}⚠ Running in CPU mode (slower)${NC}"
fi

# Start Wallie
echo ""
echo -e "${GREEN}Starting Wallie...${NC}"
echo -e "${YELLOW}Say 'Wallie' to activate!${NC}"
echo ""

# Run in foreground for testing
python wallie_voice_bot.py
