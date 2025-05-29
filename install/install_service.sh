#!/bin/bash
# Install Wallie Voice Bot as a systemd service

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get username
USER=${1:-$USER}
INSTALL_DIR="/home/$USER/wallie-voice-bot"
SERVICE_FILE="/etc/systemd/system/wallie@$USER.service"

echo -e "${GREEN}Installing Wallie Voice Bot service for user: $USER${NC}"

# Check if running as root when needed
if [ "$EUID" -ne 0 ] && [ "$1" != "$USER" ]; then 
    echo -e "${RED}Please run as root to install for another user${NC}"
    exit 1
fi

# Check if directory exists
if [ ! -d "$INSTALL_DIR" ]; then
    echo -e "${RED}Error: Wallie not found at $INSTALL_DIR${NC}"
    echo "Please clone the repository first:"
    echo "  git clone https://github.com/yourusername/wallie-voice-bot $INSTALL_DIR"
    exit 1
fi

# Check if wallie_voice_bot.py exists
if [ ! -f "$INSTALL_DIR/wallie_voice_bot.py" ]; then
    echo -e "${RED}Error: wallie_voice_bot.py not found${NC}"
    exit 1
fi

# Create config directory
CONFIG_DIR="/home/$USER/.wallie_voice_bot"
mkdir -p "$CONFIG_DIR"
chown "$USER:$USER" "$CONFIG_DIR"

# Copy example config if needed
if [ ! -f "$CONFIG_DIR/config.toml" ]; then
    if [ -f "$INSTALL_DIR/example.config.toml" ]; then
        cp "$INSTALL_DIR/example.config.toml" "$CONFIG_DIR/config.toml"
        chown "$USER:$USER" "$CONFIG_DIR/config.toml"
        echo -e "${GREEN}Created config file at $CONFIG_DIR/config.toml${NC}"
    fi
fi

# Check for PV_ACCESS_KEY
if [ -z "$PV_ACCESS_KEY" ]; then
    echo -e "${YELLOW}Warning: PV_ACCESS_KEY not set${NC}"
    echo "To enable wake word detection:"
    echo "1. Get a free API key from https://picovoice.ai"
    echo "2. Add to /etc/environment: PV_ACCESS_KEY='your-key-here'"
fi

# Install systemd service
echo -e "${GREEN}Installing systemd service...${NC}"

# Copy service file
if [ -f "$INSTALL_DIR/wallie.service" ]; then
    cp "$INSTALL_DIR/wallie.service" "$SERVICE_FILE"
    
    # Update paths in service file
    sed -i "s|/home/%i/wallie-voice-bot|$INSTALL_DIR|g" "$SERVICE_FILE"
    
    # Reload systemd
    systemctl daemon-reload
    
    echo -e "${GREEN}Service installed successfully!${NC}"
    echo
    echo "To manage the service:"
    echo "  Start:   sudo systemctl start wallie@$USER"
    echo "  Stop:    sudo systemctl stop wallie@$USER"
    echo "  Status:  sudo systemctl status wallie@$USER"
    echo "  Enable:  sudo systemctl enable wallie@$USER"
    echo "  Logs:    sudo journalctl -u wallie@$USER -f"
else
    echo -e "${RED}Error: wallie.service not found${NC}"
    exit 1
fi

# Add user to audio group
if ! groups "$USER" | grep -q audio; then
    usermod -a -G audio "$USER"
    echo -e "${GREEN}Added $USER to audio group${NC}"
    echo -e "${YELLOW}Note: User needs to log out and back in for audio group changes${NC}"
fi

# Test Python environment
echo -e "\n${GREEN}Testing Python environment...${NC}"
sudo -u "$USER" python3 "$INSTALL_DIR/test_wallie.py" || {
    echo -e "${YELLOW}Some components failed. Run setup.py to install dependencies:${NC}"
    echo "  cd $INSTALL_DIR && python3 setup.py"
}

echo -e "\n${GREEN}Installation complete!${NC}"
echo "Start Wallie with: sudo systemctl start wallie@$USER"
