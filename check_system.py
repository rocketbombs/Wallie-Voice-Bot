
#!/usr/bin/env python3
"""
System diagnostic script for Wallie Voice Bot
Checks all prerequisites before installation
"""

import sys
import os
import subprocess
import platform
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_status(message, status):
    """Print colored status message"""
    if status == "OK":
        print(f"{message}: {Colors.GREEN}✓ OK{Colors.ENDC}")
    elif status == "WARNING":
        print(f"{message}: {Colors.YELLOW}⚠ WARNING{Colors.ENDC}")
    elif status == "ERROR":
        print(f"{message}: {Colors.RED}✗ ERROR{Colors.ENDC}")
    else:
        print(f"{message}: {Colors.BLUE}{status}{Colors.ENDC}")

def check_python():
    """Check Python version"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 12:
        print_status("Python version", f"OK ({version.major}.{version.minor}.{version.micro})")
        return True
    else:
        print_status("Python version", f"ERROR ({version.major}.{version.minor}.{version.micro})")
        print("  → Python 3.12+ required")
        return False

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print_status("CUDA", f"OK ({torch.version.cuda})")
            print(f"  → GPU: {torch.cuda.get_device_name(0)}")
            print(f"  → Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            return True
        else:
            print_status("CUDA", "WARNING")
            print("  → No CUDA GPU detected, will use CPU mode")
            return False
    except ImportError:
        print_status("CUDA", "WARNING")
        print("  → PyTorch not installed yet")
        
        # Check nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print("  → NVIDIA driver detected, CUDA should work after PyTorch install")
            return False
        except:
            print("  → No NVIDIA driver found")
            return False

def check_audio():
    """Check audio system"""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        
        # Check for input devices
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        if input_devices:
            print_status("Audio Input", "OK")
            print(f"  → Found {len(input_devices)} input devices")
        else:
            print_status("Audio Input", "ERROR")
            print("  → No microphone detected!")
            return False
            
        # Check for output devices  
        output_devices = [d for d in devices if d['max_output_channels'] > 0]
        if output_devices:
            print_status("Audio Output", "OK")
            print(f"  → Found {len(output_devices)} output devices")
        else:
            print_status("Audio Output", "ERROR")
            print("  → No speakers detected!")
            return False
            
        return True
        
    except ImportError:
        print_status("Audio", "WARNING")
        print("  → sounddevice not installed yet")
        
        # Check system audio
        if platform.system() == "Linux":
            try:
                subprocess.run(['aplay', '-l'], capture_output=True, check=True)
                print("  → ALSA audio system detected")
                return True
            except:
                print("  → Audio system may need configuration")
                return False
        return True

def check_system_deps():
    """Check system dependencies"""
    if platform.system() != "Linux":
        return True
        
    missing = []
    
    # Check for required packages
    packages = {
        'portaudio19-dev': 'pkg-config --exists portaudio-2.0',
        'python3-dev': f'ls /usr/include/python{sys.version_info.major}.{sys.version_info.minor}/Python.h',
        'ffmpeg': 'which ffmpeg',
        'build-essential': 'which gcc',
    }
    
    print(f"\n{Colors.BOLD}System Dependencies:{Colors.ENDC}")
    
    for package, check_cmd in packages.items():
        try:
            subprocess.run(check_cmd.split(), capture_output=True, check=True)
            print_status(f"  {package}", "OK")
        except:
            print_status(f"  {package}", "ERROR")
            missing.append(package)
    
    if missing:
        print(f"\n{Colors.YELLOW}To install missing dependencies:{Colors.ENDC}")
        print(f"  sudo apt-get update")
        print(f"  sudo apt-get install {' '.join(missing)}")
        return False
    
    return True

def check_disk_space():
    """Check available disk space"""
    import shutil
    
    home = Path.home()
    stat = shutil.disk_usage(home)
    free_gb = stat.free / (1024**3)
    
    if free_gb >= 20:
        print_status("Disk space", f"OK ({free_gb:.1f}GB free)")
        return True
    else:
        print_status("Disk space", f"WARNING ({free_gb:.1f}GB free)")
        print("  → At least 20GB recommended for models")
        return False

def check_memory():
    """Check system memory"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        
        if total_gb >= 15.5:  # Allow for some variation in reported memory
            print_status("System RAM", f"OK ({total_gb:.1f}GB)")
            return True
        else:
            print_status("System RAM", f"WARNING ({total_gb:.1f}GB)")
            print("  → 16GB minimum required, 32GB recommended")
            return False
    except ImportError:
        # Fallback to reading /proc/meminfo
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        kb = int(line.split()[1])
                        gb = kb / (1024**2)
                        if gb >= 16:
                            print_status("System RAM", f"OK ({gb:.1f}GB)")
                        else:
                            print_status("System RAM", f"WARNING ({gb:.1f}GB)")
                        return gb >= 16
        except:
            print_status("System RAM", "WARNING")
            print("  → Could not determine memory")
            return True

def main():
    """Run all checks"""
    print(f"\n{Colors.BOLD}Wallie Voice Bot - System Check{Colors.ENDC}")
    print("=" * 50)
    
    # System info
    print(f"\n{Colors.BOLD}System Information:{Colors.ENDC}")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Architecture: {platform.machine()}")
    print(f"  Python: {platform.python_version()}")
    
    # Run checks
    print(f"\n{Colors.BOLD}Prerequisites:{Colors.ENDC}")
    
    checks = [
        ("Python Version", check_python),
        ("System Memory", check_memory),
        ("Disk Space", check_disk_space),
        ("CUDA/GPU", check_cuda),
        ("Audio System", check_audio),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print_status(name, "ERROR")
            print(f"  → Exception: {e}")
            results[name] = False
    
    # System dependencies (Linux only)
    if platform.system() == "Linux":
        results["System Deps"] = check_system_deps()
    
    # Summary
    print(f"\n{Colors.BOLD}Summary:{Colors.ENDC}")
    print("=" * 50)
    
    critical_ok = results.get("Python Version", False) and results.get("System Memory", False)
    gpu_ok = results.get("CUDA/GPU", False)
    
    if all(results.values()):
        print(f"{Colors.GREEN}✓ All checks passed! System ready for Wallie.{Colors.ENDC}")
        return 0
    elif critical_ok:
        if gpu_ok:
            print(f"{Colors.YELLOW}⚠ Some warnings, but installation should work.{Colors.ENDC}")
        else:
            print(f"{Colors.YELLOW}⚠ GPU not available. Will use CPU-only mode.{Colors.ENDC}")
            print(f"  Performance will be limited without GPU acceleration.")
        return 0
    else:
        print(f"{Colors.RED}✗ Critical requirements not met.{Colors.ENDC}")
        print(f"  Please address the errors above before installing.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
