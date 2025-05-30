#!/bin/bash
# 🐧 Wallie Voice Bot - Production Deployment Script for Linux
# Sets up Wallie as a systemd service with optimal configuration

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'
BOLD='\033[1m'

# Configuration
WALLIE_USER=${WALLIE_USER:-$USER}
WALLIE_HOME=${WALLIE_HOME:-$PWD}
SERVICE_NAME="wallie-voice-bot"

echo -e "${PURPLE}${BOLD}"
cat << "EOF"
 __      __        .__  .__  .__        
/  \    /  \_____  |  | |  | |__| ____  
\   \/\/   /\__  \ |  | |  | |  |/ __ \ 
 \        /  / __ \|  |_|  |_|  \  ___/ 
  \__/\  /  (____  /____/____/__|\___  >
       \/        \/                  \/ 
       Production Deployment - Linux
EOF
echo -e "${NC}"

echo -e "${GREEN}Deploying Wallie Voice Bot for production...${NC}"
echo ""

# Verify installation
if [ ! -f "wallie_voice_bot.py" ]; then
    echo -e "${RED}Error: wallie_voice_bot.py not found${NC}"
    echo "Please run this from the Wallie directory after installation"
    exit 1
fi

if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found${NC}"
    echo "Please run ./install_wallie.sh first"
    exit 1
fi

echo -e "${BLUE}→ Checking system requirements...${NC}"

# Check GPU
if nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo -e "  ✓ GPU: ${GREEN}$GPU_NAME${NC}"
else
    echo -e "  ${YELLOW}⚠ No GPU detected (CPU mode)${NC}"
fi

# Check audio
if command -v aplay &>/dev/null && command -v arecord &>/dev/null; then
    echo -e "  ✓ Audio system ready"
else
    echo -e "  ${RED}✗ Audio system not configured${NC}"
    exit 1
fi

# Test Wallie
echo -e "${BLUE}→ Testing Wallie installation...${NC}"
source venv/bin/activate
if python -c "import sys; sys.path.append('.'); from wallie_voice_bot import main" &>/dev/null; then
    echo -e "  ✓ Wallie imports successfully"
else
    echo -e "  ${RED}✗ Wallie import failed${NC}"
    exit 1
fi

# Create optimized config for production
echo -e "${BLUE}→ Creating production configuration...${NC}"
CONFIG_DIR="$HOME/.wallie_voice_bot"
mkdir -p "$CONFIG_DIR"/{logs,config,voices,wake_words}

# Copy Linux optimized config
if [ -f "config/linux_config.toml" ]; then
    cp config/linux_config.toml "$CONFIG_DIR/config.toml"
    echo -e "  ✓ Linux-optimized configuration applied"
else
    echo -e "  ${YELLOW}⚠ Using default configuration${NC}"
fi

# Create systemd service
echo -e "${BLUE}→ Creating systemd service...${NC}"

sudo tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null << EOF
[Unit]
Description=Wallie Voice Bot - Offline Voice Assistant
After=network.target sound.target
Wants=network.target

[Service]
Type=simple
User=$WALLIE_USER
Group=$WALLIE_USER
WorkingDirectory=$WALLIE_HOME
Environment=PATH=$WALLIE_HOME/venv/bin
Environment=PYTHONPATH=$WALLIE_HOME
Environment=CUDA_VISIBLE_DEVICES=0
Environment=OMP_NUM_THREADS=4
Environment=KMP_DUPLICATE_LIB_OK=TRUE
ExecStart=$WALLIE_HOME/venv/bin/python $WALLIE_HOME/wallie_voice_bot.py --daemon
ExecReload=/bin/kill -HUP \$MAINPID
KillMode=mixed
KillSignal=SIGTERM
TimeoutStopSec=30
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=wallie-voice-bot

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=false
ReadWritePaths=$HOME/.wallie_voice_bot
ReadWritePaths=/tmp

# Resource limits
LimitNOFILE=65536
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
EOF

echo -e "  ✓ Systemd service created"

# Create monitoring script
echo -e "${BLUE}→ Creating monitoring tools...${NC}"

cat > scripts/wallie_monitor.sh << 'EOF'
#!/bin/bash
# Wallie Voice Bot Monitor

SERVICE="wallie-voice-bot"

case "$1" in
    status)
        systemctl status $SERVICE
        ;;
    logs)
        journalctl -u $SERVICE -f
        ;;
    restart)
        sudo systemctl restart $SERVICE
        echo "Wallie restarted"
        ;;
    stop)
        sudo systemctl stop $SERVICE
        echo "Wallie stopped"
        ;;
    start)
        sudo systemctl start $SERVICE
        echo "Wallie started"
        ;;
    enable)
        sudo systemctl enable $SERVICE
        echo "Wallie enabled for auto-start"
        ;;
    disable)
        sudo systemctl disable $SERVICE
        echo "Wallie disabled from auto-start"
        ;;
    performance)
        echo "=== System Performance ==="
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
        echo ""
        echo "=== Memory Usage ==="
        free -h
        echo ""
        echo "=== CPU Usage ==="
        top -bn1 | grep "Cpu(s)"
        echo ""
        echo "=== Wallie Process ==="
        ps aux | grep wallie_voice_bot | grep -v grep
        ;;
    test)
        echo "Testing Wallie components..."
        cd "$(dirname "$0")/.."
        source venv/bin/activate
        python test_worker_startup.py
        ;;
    *)
        echo "Usage: $0 {status|logs|restart|stop|start|enable|disable|performance|test}"
        exit 1
        ;;
esac
EOF

chmod +x scripts/wallie_monitor.sh

# Create log rotation
echo -e "${BLUE}→ Setting up log rotation...${NC}"

sudo tee /etc/logrotate.d/wallie-voice-bot > /dev/null << EOF
$HOME/.wallie_voice_bot/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $WALLIE_USER $WALLIE_USER
    postrotate
        systemctl reload wallie-voice-bot > /dev/null 2>&1 || true
    endscript
}
EOF

echo -e "  ✓ Log rotation configured"

# Enable and start service
echo -e "${BLUE}→ Enabling Wallie service...${NC}"
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME

# Create desktop launcher
echo -e "${BLUE}→ Creating desktop integration...${NC}"

mkdir -p ~/.local/share/applications

cat > ~/.local/share/applications/wallie-voice-bot.desktop << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Wallie Voice Bot
Comment=Offline Voice Assistant
Exec=$WALLIE_HOME/scripts/wallie_monitor.sh status
Icon=audio-input-microphone
Terminal=true
Categories=AudioVideo;Audio;
EOF

# Final setup
echo -e "${BLUE}→ Final configuration...${NC}"

# Set audio permissions
sudo usermod -a -G audio $WALLIE_USER 2>/dev/null || true

# Create aliases
cat >> ~/.bashrc << 'EOF'

# Wallie Voice Bot aliases
alias wallie-status='sudo systemctl status wallie-voice-bot'
alias wallie-logs='journalctl -u wallie-voice-bot -f'
alias wallie-restart='sudo systemctl restart wallie-voice-bot'
alias wallie-monitor='~/$(basename "$PWD")/scripts/wallie_monitor.sh'
EOF

echo ""
echo -e "${GREEN}${BOLD}🎉 Wallie Voice Bot deployed successfully!${NC}"
echo ""
echo -e "${PURPLE}Management Commands:${NC}"
echo -e "  ${BOLD}sudo systemctl start wallie-voice-bot${NC}    - Start Wallie"
echo -e "  ${BOLD}sudo systemctl stop wallie-voice-bot${NC}     - Stop Wallie"
echo -e "  ${BOLD}sudo systemctl restart wallie-voice-bot${NC}  - Restart Wallie"
echo -e "  ${BOLD}sudo systemctl status wallie-voice-bot${NC}   - Check status"
echo -e "  ${BOLD}journalctl -u wallie-voice-bot -f${NC}        - View logs"
echo ""
echo -e "${PURPLE}Monitor Script:${NC}"
echo -e "  ${BOLD}./scripts/wallie_monitor.sh status${NC}       - Quick status"
echo -e "  ${BOLD}./scripts/wallie_monitor.sh performance${NC}  - Performance info"
echo -e "  ${BOLD}./scripts/wallie_monitor.sh test${NC}         - Test components"
echo ""
echo -e "${PURPLE}Configuration:${NC}"
echo -e "  ${BOLD}Config file:${NC} ~/.wallie_voice_bot/config.toml"
echo -e "  ${BOLD}Logs:${NC}        ~/.wallie_voice_bot/logs/"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. ${BOLD}sudo systemctl start wallie-voice-bot${NC} - Start the service"
echo -e "2. ${BOLD}./scripts/wallie_monitor.sh test${NC}      - Test functionality"
echo -e "3. Say '${BOLD}Wallie${NC}' to activate voice assistant"
echo ""
echo -e "${GREEN}Production deployment complete! ✨${NC}"
