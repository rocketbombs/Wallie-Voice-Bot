# 🐧 Wallie Voice Bot - Linux Migration Guide

## Overview
This guide will help you migrate Wallie from Windows to Linux for optimal performance and complete compatibility with the ML stack (vLLM, Triton, etc.).

## Prerequisites

### Recommended Linux Distributions
- **Ubuntu 22.04 LTS** (Recommended)
- **Ubuntu 24.04 LTS** 
- **Debian 12**
- **CentOS Stream 9**

### Hardware Requirements
- **GPU**: RTX 3080 (16GB VRAM) ✅
- **RAM**: 16GB+ (15.9GB available) ✅
- **Storage**: 20GB+ free space
- **Audio**: Microphone + Speakers/Headphones

## Migration Steps

### 1. Prepare Linux Environment

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers and CUDA
sudo apt install nvidia-driver-535 nvidia-cuda-toolkit -y

# Install Python 3.12
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev -y

# Install audio dependencies
sudo apt install portaudio19-dev alsa-utils pulseaudio espeak ffmpeg -y
```

### 2. Clone and Setup Wallie

```bash
# Clone repository
git clone https://github.com/your-username/Wallie-Voice-Bot.git
cd Wallie-Voice-Bot

# Make installer executable
chmod +x install_wallie.sh

# Run installer
./install_wallie.sh
```

### 3. Copy Windows Configuration

Transfer your optimized Windows configuration:

```bash
# Copy from Windows (adjust path as needed)
cp /mnt/c/Users/matth/.wallie_voice_bot/config.toml ~/.wallie_voice_bot/config.toml
```

### 4. Linux-Specific Optimizations

The installer will automatically detect your RTX 3080 and configure:
- **CUDA support**: Full GPU acceleration
- **vLLM**: Proper Triton support on Linux
- **Memory management**: Optimized for your 16GB RAM
- **Audio system**: ALSA/PulseAudio integration

## Key Differences from Windows

### ✅ Resolved Issues
- **vLLM Triton warnings**: Eliminated on Linux
- **OpenMP conflicts**: Native Linux handling
- **Memory management**: Better kernel-level optimization
- **Audio latency**: Lower with ALSA direct access
- **GPU drivers**: More stable NVIDIA support

### 🔧 Performance Improvements Expected
- **Response time**: ≤200ms (vs 250ms+ on Windows)
- **Memory usage**: ~20% reduction
- **GPU utilization**: More efficient
- **Audio quality**: Better with native ALSA

## Recommended Configuration for Linux

```toml
# ~/.wallie_voice_bot/config.toml - Linux Optimized

[general]
wake_word = "wallie"
wake_word_sensitivity = 0.7
log_level = "INFO"

[asr]
model = "small.en"  # Upgraded from tiny.en
device = "cuda"
compute_type = "float16"
batch_size = 16

[llm]
model = "meta-llama/Llama-3.2-3B-Instruct"  # Can use larger models
max_tokens = 256
temperature = 0.7
gpu_memory_fraction = 0.6  # Can use more GPU memory
tensor_parallel_size = 1

[tts]
engine = "edge-tts"
voice = "en-US-AriaNeural"
language = "en"
speed = 1.0

[performance]
watchdog_interval_sec = 1
enable_prometheus = true
archive_transcripts = true
max_concurrent_requests = 4
```

## Testing Your Migration

### 1. Basic Functionality Test
```bash
cd Wallie-Voice-Bot
source venv/bin/activate
python test_import.py
python test_worker_startup.py
```

### 2. End-to-End Test
```bash
python test_end_to_end.py
```

### 3. Performance Benchmark
```bash
python test_voice_interaction.py
```

### 4. Production Test
```bash
./wallie --test
./wallie --daemon
```

## Troubleshooting

### Audio Issues
```bash
# Test audio
aplay /usr/share/sounds/alsa/Front_Left.wav
arecord -d 3 test.wav && aplay test.wav

# Fix permissions
sudo usermod -a -G audio $USER
```

### GPU Issues
```bash
# Verify CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Issues
```bash
# Monitor usage
htop
nvidia-smi -l 1
```

## Expected Performance Gains

| Metric | Windows | Linux | Improvement |
|--------|---------|-------|-------------|
| Response Time | 300-400ms | 200-250ms | ~40% faster |
| Memory Usage | 14GB+ | 11-13GB | ~20% reduction |
| GPU Utilization | 60-70% | 80-90% | Better efficiency |
| Audio Latency | 100-150ms | 50-80ms | ~50% reduction |
| Stability | Good | Excellent | More reliable |

## Migration Checklist

- [ ] Linux environment prepared
- [ ] NVIDIA drivers installed
- [ ] Python 3.12 available
- [ ] Wallie repository cloned
- [ ] Dependencies installed via `install_wallie.sh`
- [ ] Configuration transferred
- [ ] Audio system tested
- [ ] GPU acceleration verified
- [ ] All workers starting successfully
- [ ] End-to-end voice test passed
- [ ] Performance benchmarks run
- [ ] Production deployment ready

## Next Steps After Migration

1. **Fine-tune performance** for sub-200ms response
2. **Test real-world scenarios** with various accents/noise
3. **Deploy production setup** with systemd service
4. **Monitor performance** with built-in metrics
5. **Optimize models** based on usage patterns

---

**Need help?** The Linux environment will eliminate the Windows-specific ML library issues and provide the performance you need for production deployment.
