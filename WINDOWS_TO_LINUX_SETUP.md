# 🪟➡️🐧 Windows to Linux Setup Guide for Wallie

## Overview
You're currently on Windows but need Linux to run Wallie optimally. Here are your options to get a Linux environment with GPU access for Wallie.

## Option 1: WSL2 with GPU Support (Recommended for Testing) ⭐

### Advantages
- ✅ Keep Windows as primary OS
- ✅ Native GPU passthrough support
- ✅ Easy file sharing between Windows/Linux
- ✅ Good for development and testing

### Setup Steps

#### 1. Enable WSL2
```powershell
# Run in PowerShell as Administrator
wsl --install
# Restart computer when prompted
```

#### 2. Install Ubuntu 22.04
```powershell
# After restart, install Ubuntu
wsl --install -d Ubuntu-22.04
```

#### 3. Setup NVIDIA GPU Support in WSL2
```powershell
# Download and install NVIDIA drivers for WSL2
# Go to: https://developer.nvidia.com/cuda/wsl
# Install the Windows driver that includes WSL2 support
```

#### 4. Enter WSL2 and Setup Linux Environment
```bash
# Enter WSL2 Ubuntu
wsl

# Update system
sudo apt update && sudo apt upgrade -y

# Install CUDA toolkit for WSL2
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda

# Install Python 3.12
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev -y

# Install audio dependencies (limited in WSL2)
sudo apt install portaudio19-dev build-essential -y
```

#### 5. Clone Wallie in WSL2
```bash
# Access your Windows files from WSL2
cd /mnt/c/Users/matth/OneDrive/Documents/GitHub/Wallie-Voice-Bot

# Or clone fresh in WSL2
git clone https://github.com/your-username/Wallie-Voice-Bot.git ~/wallie
cd ~/wallie
```

#### 6. Run Wallie Installation
```bash
chmod +x install_wallie.sh
./install_wallie.sh
```

### ⚠️ WSL2 Limitations
- Audio system is limited (no microphone/speaker access)
- Good for development but not full voice interaction
- GPU works but may have some performance overhead

---

## Option 2: Dual Boot Ubuntu (Recommended for Production) 🏆

### Advantages
- ✅ Best performance (native Linux)
- ✅ Full hardware access (audio, GPU)
- ✅ Production-ready environment
- ✅ No virtualization overhead

### Setup Steps

#### 1. Create Ubuntu USB
1. Download Ubuntu 22.04 LTS from https://ubuntu.com/download/desktop
2. Use Rufus or similar to create bootable USB
3. Boot from USB and select "Install Ubuntu alongside Windows"

#### 2. Partition Recommendations
- **Ubuntu**: 60GB+ (for Wallie + models)
- **Swap**: 16GB (same as RAM)
- **Keep Windows partition** for dual boot

#### 3. After Ubuntu Installation
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers
sudo ubuntu-drivers autoinstall
sudo reboot

# Verify GPU
nvidia-smi
```

#### 4. Install Wallie
```bash
# Clone Wallie
git clone https://github.com/your-username/Wallie-Voice-Bot.git ~/wallie
cd ~/wallie

# Run installation
chmod +x install_wallie.sh
./install_wallie.sh

# Deploy production
chmod +x deploy_linux.sh
./deploy_linux.sh
```

---

## Option 3: VirtualBox/VMware (Not Recommended)

### Why Not Recommended
- ❌ No GPU passthrough (without complex setup)
- ❌ Poor performance for ML workloads
- ❌ Audio issues
- ❌ Not suitable for Wallie's requirements

---

## Option 4: Cloud Linux Instance (For Testing Only)

### Providers with GPU Support
- **Google Cloud**: A100/V100 instances
- **AWS EC2**: P3/P4 instances  
- **Azure**: NCv3 series

### Considerations
- 💰 Expensive for continuous use
- 🌐 Network latency affects voice interaction
- 🎤 No local audio (testing only)

---

## Recommendation for Your Setup

Given that you have an RTX 3080 and want optimal performance:

### For Development/Testing: WSL2
```powershell
# Quick setup in PowerShell (as Administrator)
wsl --install -d Ubuntu-22.04
# Restart, then follow WSL2 steps above
```

### For Production: Dual Boot Ubuntu
1. **Backup important Windows data**
2. **Create Ubuntu installation USB**
3. **Install Ubuntu alongside Windows**
4. **Set up Wallie natively on Ubuntu**

## Quick WSL2 Setup Script

Let me create a PowerShell script to set up WSL2 for you:

```powershell
# Check if WSL is available
if (Get-Command wsl -ErrorAction SilentlyContinue) {
    Write-Host "WSL is available" -ForegroundColor Green
    wsl --status
} else {
    Write-Host "Installing WSL..." -ForegroundColor Yellow
    wsl --install
    Write-Host "Please restart your computer and run this script again" -ForegroundColor Yellow
    exit
}

# Install Ubuntu if not present
$distros = wsl --list --quiet
if ($distros -notcontains "Ubuntu-22.04") {
    Write-Host "Installing Ubuntu 22.04..." -ForegroundColor Yellow
    wsl --install -d Ubuntu-22.04
} else {
    Write-Host "Ubuntu 22.04 already installed" -ForegroundColor Green
}

Write-Host @"
Next steps:
1. wsl  # Enter Ubuntu
2. Follow the Linux setup commands from the guide
3. cd /mnt/c/Users/matth/OneDrive/Documents/GitHub/Wallie-Voice-Bot
4. ./install_wallie.sh
"@ -ForegroundColor Cyan
```

## Which Option Should You Choose?

**For immediate testing**: Start with WSL2
**For production Wallie**: Dual boot Ubuntu

Would you like me to help you set up WSL2 first to test Wallie, or do you prefer to go straight to dual boot Ubuntu?
