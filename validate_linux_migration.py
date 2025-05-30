#!/usr/bin/env python3
"""
🐧 Wallie Voice Bot - Linux Migration Validation
Comprehensive test suite to validate Linux deployment
"""

import os
import sys
import time
import subprocess
import importlib
import signal
from pathlib import Path
import json

class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    NC = '\033[0m'
    BOLD = '\033[1m'

def print_banner():
    print(f"{Colors.PURPLE}{Colors.BOLD}")
    print("""
 __      __        .__  .__  .__        
/  \    /  \_____  |  | |  | |__| ____  
\   \/\/   /\__  \ |  | |  | |  |/ __ \ 
 \        /  / __ \|  |_|  |_|  \  ___/ 
  \__/\  /  (____  /____/____/__|\___  >
       \/        \/                  \/ 
       Linux Migration Validation
    """)
    print(f"{Colors.NC}")

def run_command(cmd, capture_output=True, timeout=30):
    """Run shell command with timeout"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=capture_output, 
            text=True, timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def test_system_requirements():
    """Test Linux system requirements"""
    print(f"{Colors.BLUE}→ Testing system requirements...{Colors.NC}")
    
    tests = []
    
    # Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        tests.append(("Python version", True, f"{python_version.major}.{python_version.minor}"))
    else:
        tests.append(("Python version", False, f"{python_version.major}.{python_version.minor} (need 3.8+)"))
    
    # CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            tests.append(("CUDA GPU", True, gpu_name))
        else:
            tests.append(("CUDA GPU", False, "Not available"))
    except ImportError:
        tests.append(("CUDA GPU", False, "PyTorch not installed"))
    
    # NVIDIA driver
    success, stdout, _ = run_command("nvidia-smi --query-gpu=name --format=csv,noheader")
    if success and stdout.strip():
        tests.append(("NVIDIA Driver", True, stdout.strip()))
    else:
        tests.append(("NVIDIA Driver", False, "Not available"))
    
    # Audio system
    success, _, _ = run_command("which aplay")
    if success:
        tests.append(("Audio System", True, "ALSA available"))
    else:
        tests.append(("Audio System", False, "ALSA not found"))
    
    # Memory
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemTotal:' in line:
                    mem_kb = int(line.split()[1])
                    mem_gb = mem_kb / 1024 / 1024
                    if mem_gb >= 15:
                        tests.append(("System Memory", True, f"{mem_gb:.1f}GB"))
                    else:
                        tests.append(("System Memory", False, f"{mem_gb:.1f}GB (need 16GB+)"))
                    break
    except:
        tests.append(("System Memory", False, "Cannot detect"))
    
    # Print results
    for test_name, passed, details in tests:
        status = f"{Colors.GREEN}✓{Colors.NC}" if passed else f"{Colors.RED}✗{Colors.NC}"
        print(f"  {status} {test_name}: {details}")
    
    return all(test[1] for test in tests)

def test_python_imports():
    """Test all required Python imports"""
    print(f"{Colors.BLUE}→ Testing Python imports...{Colors.NC}")
    
    required_modules = [
        ('torch', 'PyTorch'),
        ('faster_whisper', 'Faster Whisper'),
        ('pvporcupine', 'Porcupine Wake Word'),
        ('pyttsx3', 'Text-to-Speech'),
        ('edge_tts', 'Edge TTS'),
        ('sounddevice', 'Audio I/O'),
        ('numpy', 'NumPy'),
        ('toml', 'TOML Config'),
        ('psutil', 'System Utils'),
    ]
    
    # Try vLLM import (should work on Linux)
    try:
        import vllm
        required_modules.append(('vllm', 'vLLM (Linux)'))
    except ImportError:
        print(f"  {Colors.YELLOW}⚠{Colors.NC} vLLM: Not available (fallback to transformers)")
    
    failed_imports = []
    for module_name, display_name in required_modules:
        try:
            importlib.import_module(module_name)
            print(f"  {Colors.GREEN}✓{Colors.NC} {display_name}")
        except ImportError as e:
            print(f"  {Colors.RED}✗{Colors.NC} {display_name}: {e}")
            failed_imports.append(module_name)
    
    return len(failed_imports) == 0

def test_wallie_components():
    """Test Wallie components"""
    print(f"{Colors.BLUE}→ Testing Wallie components...{Colors.NC}")
    
    # Test main module import
    try:
        sys.path.insert(0, '.')
        import wallie_voice_bot
        print(f"  {Colors.GREEN}✓{Colors.NC} Main module import")
        main_import = True
    except Exception as e:
        print(f"  {Colors.RED}✗{Colors.NC} Main module import: {e}")
        main_import = False
    
    # Test worker imports
    worker_imports = {}
    workers = ['vad_worker', 'asr_worker', 'llm_worker', 'tts_worker']
    
    for worker in workers:
        try:
            module = importlib.import_module(f'workers.{worker}')
            print(f"  {Colors.GREEN}✓{Colors.NC} {worker}")
            worker_imports[worker] = True
        except Exception as e:
            print(f"  {Colors.RED}✗{Colors.NC} {worker}: {e}")
            worker_imports[worker] = False
    
    return main_import and all(worker_imports.values())

def test_configuration():
    """Test configuration files"""
    print(f"{Colors.BLUE}→ Testing configuration...{Colors.NC}")
    
    config_dir = Path.home() / '.wallie_voice_bot'
    config_file = config_dir / 'config.toml'
    
    tests = []
    
    # Config directory
    if config_dir.exists():
        tests.append(("Config directory", True, str(config_dir)))
    else:
        tests.append(("Config directory", False, "Missing"))
    
    # Config file
    if config_file.exists():
        try:
            import toml
            config = toml.load(config_file)
            tests.append(("Config file", True, "Valid TOML"))
            
            # Check key sections
            required_sections = ['asr', 'llm', 'tts']
            for section in required_sections:
                if section in config:
                    tests.append((f"Config [{section}]", True, "Present"))
                else:
                    tests.append((f"Config [{section}]", False, "Missing"))
        except Exception as e:
            tests.append(("Config file", False, f"Invalid: {e}"))
    else:
        tests.append(("Config file", False, "Missing"))
    
    # Print results
    for test_name, passed, details in tests:
        status = f"{Colors.GREEN}✓{Colors.NC}" if passed else f"{Colors.RED}✗{Colors.NC}"
        print(f"  {status} {test_name}: {details}")
    
    return all(test[1] for test in tests)

def test_audio_system():
    """Test Linux audio system"""
    print(f"{Colors.BLUE}→ Testing audio system...{Colors.NC}")
    
    tests = []
    
    # ALSA
    success, _, _ = run_command("which aplay")
    tests.append(("ALSA playback", success, "Available" if success else "Missing"))
    
    success, _, _ = run_command("which arecord")
    tests.append(("ALSA recording", success, "Available" if success else "Missing"))
    
    # PulseAudio
    success, _, _ = run_command("which pulseaudio")
    tests.append(("PulseAudio", success, "Available" if success else "Missing"))
    
    # List audio devices
    success, stdout, _ = run_command("aplay -l")
    if success and "card" in stdout:
        tests.append(("Audio devices", True, "Detected"))
    else:
        tests.append(("Audio devices", False, "None found"))
    
    # Test sounddevice
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        output_devices = [d for d in devices if d['max_output_channels'] > 0]
        
        if input_devices:
            tests.append(("Input devices", True, f"{len(input_devices)} found"))
        else:
            tests.append(("Input devices", False, "None found"))
            
        if output_devices:
            tests.append(("Output devices", True, f"{len(output_devices)} found"))
        else:
            tests.append(("Output devices", False, "None found"))
            
    except Exception as e:
        tests.append(("SoundDevice", False, str(e)))
    
    # Print results
    for test_name, passed, details in tests:
        status = f"{Colors.GREEN}✓{Colors.NC}" if passed else f"{Colors.RED}✗{Colors.NC}"
        print(f"  {status} {test_name}: {details}")
    
    return all(test[1] for test in tests)

def test_performance_benchmarks():
    """Run performance benchmarks"""
    print(f"{Colors.BLUE}→ Running performance benchmarks...{Colors.NC}")
    
    benchmarks = []
    
    # PyTorch GPU benchmark
    try:
        import torch
        if torch.cuda.is_available():
            # GPU memory test
            device = torch.device('cuda')
            start_time = time.time()
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.mm(x, y)
            torch.cuda.synchronize()
            gpu_time = (time.time() - start_time) * 1000
            benchmarks.append(("GPU Matrix Multiply", True, f"{gpu_time:.1f}ms"))
            
            # GPU memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            used_memory = torch.cuda.memory_allocated(0) / 1024**3
            benchmarks.append(("GPU Memory", True, f"{used_memory:.1f}GB / {total_memory:.1f}GB"))
        else:
            benchmarks.append(("GPU Benchmark", False, "CUDA not available"))
    except Exception as e:
        benchmarks.append(("GPU Benchmark", False, str(e)))
    
    # CPU benchmark
    try:
        import numpy as np
        start_time = time.time()
        a = np.random.randn(1000, 1000)
        b = np.random.randn(1000, 1000)
        c = np.dot(a, b)
        cpu_time = (time.time() - start_time) * 1000
        benchmarks.append(("CPU Matrix Multiply", True, f"{cpu_time:.1f}ms"))
    except Exception as e:
        benchmarks.append(("CPU Benchmark", False, str(e)))
    
    # Memory usage
    try:
        import psutil
        memory = psutil.virtual_memory()
        benchmarks.append(("System Memory Usage", True, f"{memory.percent:.1f}% ({memory.used/1024**3:.1f}GB used)"))
    except Exception as e:
        benchmarks.append(("Memory Usage", False, str(e)))
    
    # Print results
    for test_name, passed, details in tests:
        status = f"{Colors.GREEN}✓{Colors.NC}" if passed else f"{Colors.RED}✗{Colors.NC}"
        print(f"  {status} {test_name}: {details}")
    
    return True

def test_worker_startup():
    """Test worker startup (quick test)"""
    print(f"{Colors.BLUE}→ Testing worker startup...{Colors.NC}")
    
    # This would be more comprehensive, but for migration validation
    # we'll do a quick import test
    try:
        from workers.worker_base import WorkerBase
        print(f"  {Colors.GREEN}✓{Colors.NC} Worker base class")
        
        workers = ['VADWorker', 'ASRWorker', 'LLMWorker', 'TTSWorker']
        for worker_name in workers:
            try:
                if worker_name == 'VADWorker':
                    from workers.vad_worker import VADWorker
                elif worker_name == 'ASRWorker':
                    from workers.asr_worker import ASRWorker
                elif worker_name == 'LLMWorker':
                    from workers.llm_worker import LLMWorker
                elif worker_name == 'TTSWorker':
                    from workers.tts_worker import TTSWorker
                print(f"  {Colors.GREEN}✓{Colors.NC} {worker_name}")
            except Exception as e:
                print(f"  {Colors.RED}✗{Colors.NC} {worker_name}: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"  {Colors.RED}✗{Colors.NC} Worker base: {e}")
        return False

def generate_report(results):
    """Generate migration validation report"""
    print(f"\n{Colors.PURPLE}{Colors.BOLD}📊 Migration Validation Report{Colors.NC}")
    print("=" * 50)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {Colors.GREEN}{passed_tests}{Colors.NC}")
    print(f"Failed: {Colors.RED}{total_tests - passed_tests}{Colors.NC}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print("\nTest Results:")
    for test_name, passed in results.items():
        status = f"{Colors.GREEN}PASS{Colors.NC}" if passed else f"{Colors.RED}FAIL{Colors.NC}"
        print(f"  {status} {test_name}")
    
    if all(results.values()):
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 All tests passed! Linux migration successful!{Colors.NC}")
        print(f"{Colors.GREEN}✅ Wallie is ready for production on Linux{Colors.NC}")
        return True
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠ Some tests failed{Colors.NC}")
        failed_tests = [name for name, passed in results.items() if not passed]
        print(f"{Colors.YELLOW}Failed tests: {', '.join(failed_tests)}{Colors.NC}")
        return False

def main():
    """Run complete migration validation"""
    print_banner()
    print(f"{Colors.GREEN}Validating Wallie Voice Bot Linux migration...{Colors.NC}\n")
    
    # Run all tests
    results = {}
    
    results["System Requirements"] = test_system_requirements()
    print()
    
    results["Python Imports"] = test_python_imports()
    print()
    
    results["Wallie Components"] = test_wallie_components()
    print()
    
    results["Configuration"] = test_configuration()
    print()
    
    results["Audio System"] = test_audio_system()
    print()
    
    results["Performance"] = test_performance_benchmarks()
    print()
    
    results["Worker Startup"] = test_worker_startup()
    print()
    
    # Generate report
    success = generate_report(results)
    
    if success:
        print(f"\n{Colors.GREEN}Next steps:{Colors.NC}")
        print("1. Run: sudo systemctl start wallie-voice-bot")
        print("2. Test: ./scripts/wallie_monitor.sh test")
        print("3. Say 'Wallie' to activate!")
        sys.exit(0)
    else:
        print(f"\n{Colors.YELLOW}Please fix failing tests before deployment{Colors.NC}")
        sys.exit(1)

if __name__ == "__main__":
    main()
