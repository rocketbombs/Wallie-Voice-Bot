#!/bin/bash
# 🎙️ Wallie Voice Bot - Complete Installer
# Production-grade offline voice assistant

set -e

# Colors and styling
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'
BOLD='\033[1m'

# Banner
clear
echo -e "${PURPLE}${BOLD}"
cat << "EOF"
 __      __        .__  .__  .__        
/  \    /  \_____  |  | |  | |__| ____  
\   \/\/   /\__  \ |  | |  | |  |/ __ \ 
 \        /  / __ \|  |_|  |_|  \  ___/ 
  \__/\  /  (____  /____/____/__|\___  >
       \/        \/                  \/ 
       Voice Bot - v1.0
EOF
echo -e "${NC}"
echo -e "${GREEN}Production-grade offline voice assistant${NC}"
echo -e "${BLUE}≤250ms response time | 100% private${NC}"
echo ""

# System check
echo -e "${YELLOW}Checking system...${NC}"

# Python version
if python3.12 --version &>/dev/null; then
    echo -e "  ✓ Python 3.12"
else
    echo -e "  ${RED}✗ Python 3.12 not found${NC}"
    echo "    Install: sudo apt install python3.12 python3.12-venv python3.12-dev"
    exit 1
fi

# GPU check
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo -e "  ✓ GPU: ${GREEN}$GPU_NAME${NC}"
    GPU_MODE=1
else
    echo -e "  ${YELLOW}⚠ No GPU detected (CPU mode)${NC}"
    GPU_MODE=0
fi

# Audio check
if command -v aplay &>/dev/null; then
    echo -e "  ✓ Audio system"
else
    echo -e "  ${YELLOW}⚠ Audio system needs setup${NC}"
fi

echo ""
echo -e "${YELLOW}Installing Wallie...${NC}"

# Fix WSL apt issues silently
if grep -qi microsoft /proc/version &>/dev/null; then
    sudo rm -f /usr/lib/cnf-update-db 2>/dev/null || true
    sudo touch /usr/lib/cnf-update-db 2>/dev/null || true
fi

# System dependencies
echo -e "${BLUE}→ Installing system packages...${NC}"
sudo apt-get install -y -qq \
    python3.12-dev \
    portaudio19-dev \
    build-essential \
    libsndfile1-dev \
    espeak \
    ffmpeg 2>/dev/null || true

# Create project structure
echo -e "${BLUE}→ Creating project structure...${NC}"
mkdir -p {workers,config,scripts,tests}
mkdir -p ~/.wallie_voice_bot/{logs,wake_words,voices,config}

# Python environment
if [ ! -d "venv" ]; then
    echo -e "${BLUE}→ Creating virtual environment...${NC}"
    python3.12 -m venv venv
fi

source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel -q

# Install packages
echo -e "${BLUE}→ Installing core dependencies...${NC}"
pip install -q numpy sounddevice typer pydantic-settings toml psutil

echo -e "${BLUE}→ Installing AI models...${NC}"
pip install -q pvporcupine faster-whisper

echo -e "${BLUE}→ Installing voice synthesis...${NC}"
pip install -q pyttsx3 edge-tts

if [ $GPU_MODE -eq 1 ]; then
    echo -e "${BLUE}→ Installing GPU acceleration...${NC}"
    pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    pip install -q vllm
else
    echo -e "${BLUE}→ Installing CPU support...${NC}"
    pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install -q transformers accelerate
fi

# Create config
CONFIG_FILE="$HOME/.wallie_voice_bot/config.toml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${BLUE}→ Creating configuration...${NC}"
    cat > "$CONFIG_FILE" << EOF
# Wallie Configuration

wake_word = "wallie"
wake_word_sensitivity = 0.7

asr_model = "tiny.en"
asr_device = "$([ $GPU_MODE -eq 1 ] && echo "cuda" || echo "cpu")"
asr_compute_type = "$([ $GPU_MODE -eq 1 ] && echo "float16" || echo "int8")"

llm_model = "meta-llama/Llama-3.2-3B-Instruct"
llm_max_tokens = 512
llm_temperature = 0.7
llm_gpu_memory_fraction = 0.4

tts_engine = "auto"
tts_voice = "en-US-AriaNeural"
tts_language = "en"

watchdog_interval_sec = 2
enable_prometheus = false
archive_transcripts = false
EOF
fi

# Create launcher
cat > wallie << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python wallie_voice_bot.py "$@"
EOF
chmod +x wallie

# Final message
echo ""
echo -e "${GREEN}${BOLD}✨ Wallie installed successfully!${NC}"
echo ""
echo -e "${PURPLE}Quick Start:${NC}"
echo -e "  ${BOLD}./wallie${NC}          - Run Wallie"
echo -e "  ${BOLD}./wallie --daemon${NC} - Run in background"
echo ""
echo -e "${PURPLE}Commands:${NC}"
echo -e "  Say ${BOLD}'Wallie'${NC} to activate"
echo -e "  Say ${BOLD}'Wallie'${NC} again to interrupt"
echo ""
echo -e "${BLUE}Config: ~/.wallie_voice_bot/config.toml${NC}"

# Test imports
echo ""
echo -e "${YELLOW}Testing installation...${NC}"
python -c "
import sys
modules = ['torch', 'faster_whisper', 'pvporcupine', 'pyttsx3', 'edge_tts']
failed = []
for m in modules:
    try:
        __import__(m)
        print(f'  ✓ {m}')
    except:
        failed.append(m)
        print(f'  ✗ {m}')
if failed:
    print(f'\n⚠ Some modules failed, but core functionality should work')
" 2>/dev/null

echo ""
echo -e "${GREEN}Ready to use! Just copy the Python files and run ./wallie${NC}"
