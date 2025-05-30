# 🪟🐧 Wallie WSL2 Setup Script
# Sets up Windows Subsystem for Linux 2 with GPU support for Wallie

param(
    [switch]$Force,
    [switch]$SkipGPU,
    [string]$Distribution = "Ubuntu-22.04"
)

# Colors for output
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Test-Administrator {
    $currentUser = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
    return $currentUser.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

Write-ColorOutput @"
 __      __        .__  .__  .__        
/  \    /  \_____  |  | |  | |__| ____  
\   \/\/   /\__  \ |  | |  | |  |/ __ \ 
 \        /  / __ \|  |_|  |_|  \  ___/ 
  \__/\  /  (____  /____/____/__|\___  >
       \/        \/                  \/ 
       WSL2 Setup for Windows
"@ -Color Magenta

Write-ColorOutput "Setting up Linux environment for Wallie Voice Bot..." -Color Green
Write-Host ""

# Check if running as administrator
if (-not (Test-Administrator)) {
    Write-ColorOutput "❌ This script requires administrator privileges!" -Color Red
    Write-ColorOutput "Please run PowerShell as Administrator and try again." -Color Yellow
    exit 1
}

# Check Windows version
$winVersion = [System.Environment]::OSVersion.Version
if ($winVersion.Build -lt 19041) {
    Write-ColorOutput "❌ WSL2 requires Windows 10 version 2004 or later (build 19041+)" -Color Red
    Write-ColorOutput "Current build: $($winVersion.Build)" -Color Yellow
    exit 1
}

Write-ColorOutput "✅ Windows version compatible (build $($winVersion.Build))" -Color Green

# Check if WSL is available
Write-ColorOutput "🔍 Checking WSL availability..." -Color Blue

try {
    $wslStatus = wsl --status 2>$null
    $wslAvailable = $?
} catch {
    $wslAvailable = $false
}

if (-not $wslAvailable) {
    Write-ColorOutput "📦 Installing WSL..." -Color Yellow
    
    # Enable WSL and Virtual Machine Platform
    dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
    dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
    
    # Download and install WSL2 kernel update
    $kernelUrl = "https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi"
    $kernelPath = "$env:TEMP\wsl_update_x64.msi"
    
    Write-ColorOutput "⬇️ Downloading WSL2 kernel update..." -Color Blue
    Invoke-WebRequest -Uri $kernelUrl -OutFile $kernelPath
    
    Write-ColorOutput "📦 Installing WSL2 kernel update..." -Color Blue
    Start-Process msiexec.exe -ArgumentList "/i", $kernelPath, "/quiet" -Wait
    
    # Set WSL2 as default
    wsl --set-default-version 2
    
    Write-ColorOutput "🔄 WSL installed. A restart is required." -Color Yellow
    Write-ColorOutput "After restart, run this script again to continue setup." -Color Cyan
    
    $restart = Read-Host "Restart now? (y/n)"
    if ($restart -eq 'y' -or $restart -eq 'Y') {
        Restart-Computer -Force
    }
    exit 0
}

Write-ColorOutput "✅ WSL2 is available" -Color Green

# Check installed distributions
Write-ColorOutput "🔍 Checking installed Linux distributions..." -Color Blue
$distributions = wsl --list --quiet 2>$null | Where-Object { $_ -ne "" }

if ($distributions -contains $Distribution) {
    Write-ColorOutput "✅ $Distribution already installed" -Color Green
    $installDistro = $false
} else {
    Write-ColorOutput "📦 Installing $Distribution..." -Color Yellow
    $installDistro = $true
}

if ($installDistro -or $Force) {
    try {
        wsl --install -d $Distribution
        Write-ColorOutput "✅ $Distribution installed successfully" -Color Green
    } catch {
        Write-ColorOutput "❌ Failed to install $Distribution" -Color Red
        Write-ColorOutput "You may need to install it manually from Microsoft Store" -Color Yellow
        exit 1
    }
}

# Check GPU support
if (-not $SkipGPU) {
    Write-ColorOutput "🎮 Checking GPU support..." -Color Blue
    
    try {
        $gpuInfo = nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
        if ($?) {
            Write-ColorOutput "✅ NVIDIA GPU detected: $gpuInfo" -Color Green
            Write-ColorOutput "💡 Make sure you have NVIDIA drivers that support WSL2" -Color Cyan
            Write-ColorOutput "   Download from: https://developer.nvidia.com/cuda/wsl" -Color Cyan
        } else {
            Write-ColorOutput "⚠️ NVIDIA GPU not detected or drivers not installed" -Color Yellow
        }
    } catch {
        Write-ColorOutput "⚠️ Could not check GPU status" -Color Yellow
    }
}

# Create setup script for Linux side
$linuxSetupScript = @"
#!/bin/bash
# WSL2 Linux setup script for Wallie

echo "🐧 Setting up Linux environment for Wallie..."

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python 3.12
echo "🐍 Installing Python 3.12..."
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev python3.12-distutils -y

# Install build tools
echo "🔨 Installing build tools..."
sudo apt install build-essential pkg-config -y

# Install audio dependencies (limited in WSL2)
echo "🎵 Installing audio dependencies..."
sudo apt install portaudio19-dev libsndfile1-dev espeak -y

# Install GPU support if available
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 Setting up CUDA for WSL2..."
    
    # Add NVIDIA package repository
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update
    
    # Install CUDA toolkit
    sudo apt-get -y install cuda-toolkit-12-4
    
    # Add to PATH
    echo 'export PATH=/usr/local/cuda/bin:\$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH' >> ~/.bashrc
    
    echo "✅ CUDA toolkit installed"
else
    echo "⚠️ No NVIDIA GPU detected in WSL2"
fi

# Setup Wallie
echo "🎙️ Setting up Wallie..."
WALLIE_DIR="/mnt/c/Users/$USER/OneDrive/Documents/GitHub/Wallie-Voice-Bot"
if [ -d "\$WALLIE_DIR" ]; then
    echo "📁 Found Wallie directory at \$WALLIE_DIR"
    cd "\$WALLIE_DIR"
    
    if [ -f "install_wallie.sh" ]; then
        echo "🚀 Running Wallie installer..."
        chmod +x install_wallie.sh
        ./install_wallie.sh
        
        echo ""
        echo "✅ WSL2 setup complete!"
        echo ""
        echo "🎯 Next steps:"
        echo "1. Test: python validate_linux_migration.py"
        echo "2. Run: ./wallie"
        echo ""
        echo "⚠️ Note: Audio will be limited in WSL2"
        echo "   For full audio support, consider dual-boot Ubuntu"
    else
        echo "❌ install_wallie.sh not found in \$WALLIE_DIR"
    fi
else
    echo "❌ Wallie directory not found at \$WALLIE_DIR"
    echo "Please make sure Wallie is in your Windows Documents/GitHub folder"
fi

echo ""
echo "🐧 Linux environment ready for Wallie development!"
"@

# Save the Linux setup script
$linuxScriptPath = "$PWD\wsl_setup_linux.sh"
$linuxSetupScript | Out-File -FilePath $linuxScriptPath -Encoding UTF8

Write-ColorOutput ""
Write-ColorOutput "🎉 WSL2 setup complete!" -Color Green
Write-ColorOutput ""
Write-ColorOutput "📋 Next steps:" -Color Cyan
Write-ColorOutput "1. wsl                           # Enter Ubuntu" -Color White
Write-ColorOutput "2. bash /mnt/c$(($linuxScriptPath).Replace('\', '/').Replace('C:', ''))  # Run Linux setup" -Color White
Write-ColorOutput ""
Write-ColorOutput "🚀 Quick start:" -Color Cyan
Write-ColorOutput "wsl -e bash /mnt/c$((Get-Location).Path.Replace('\', '/').Replace('C:', ''))/wsl_setup_linux.sh" -Color White
Write-ColorOutput ""
Write-ColorOutput "⚠️ WSL2 Limitations for Wallie:" -Color Yellow
Write-ColorOutput "• Limited audio support (no microphone/speakers)" -Color White
Write-ColorOutput "• Good for development and testing" -Color White
Write-ColorOutput "• For production, consider dual-boot Ubuntu" -Color White
Write-ColorOutput ""
Write-ColorOutput "💡 For full Wallie experience, see WINDOWS_TO_LINUX_SETUP.md" -Color Cyan

# Option to run Linux setup immediately
$runNow = Read-Host "Run Linux setup now? (y/n)"
if ($runNow -eq 'y' -or $runNow -eq 'Y') {
    Write-ColorOutput "🚀 Starting Linux setup..." -Color Green
    $scriptPath = "/mnt/c$((Get-Location).Path.Replace('\', '/').Replace('C:', ''))/wsl_setup_linux.sh"
    wsl -e bash $scriptPath
}
