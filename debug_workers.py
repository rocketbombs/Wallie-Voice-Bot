#!/usr/bin/env python3
"""Debug worker startup issues"""

import os
import sys
from pathlib import Path

print("🔍 Wallie Worker Diagnostics\n")

# Check Python version
print(f"Python: {sys.version}")

# Check environment
print(f"\n✓ PV_ACCESS_KEY: {'Set' if os.environ.get('PV_ACCESS_KEY') else '❌ NOT SET'}")

# Test imports
print("\nTesting imports:")
modules = {
    'numpy': 'Core',
    'sounddevice': 'Audio',
    'pvporcupine': 'Wake word',
    'faster_whisper': 'ASR', 
    'torch': 'PyTorch',
    'vllm': 'LLM (optional)',
    'pyttsx3': 'TTS',
    'edge_tts': 'TTS (online)'
}

for module, desc in modules.items():
    try:
        __import__(module)
        print(f"  ✓ {module:<15} ({desc})")
    except ImportError as e:
        print(f"  ❌ {module:<15} ({desc}): {str(e)[:50]}")

# Test Porcupine
print("\nTesting Porcupine:")
try:
    import pvporcupine
    porcupine = pvporcupine.create(
        access_key=os.environ.get('PV_ACCESS_KEY', ''),
        keywords=['picovoice']
    )
    print("  ✓ Porcupine initialized")
    porcupine.delete()
except Exception as e:
    print(f"  ❌ Porcupine error: {e}")

# Test audio
print("\nTesting audio:")
try:
    import sounddevice as sd
    devices = sd.query_devices()
    print(f"  ✓ Found {len(devices)} audio devices")
    print(f"  ✓ Default input: {sd.query_devices(kind='input')['name']}")
    print(f"  ✓ Default output: {sd.query_devices(kind='output')['name']}")
except Exception as e:
    print(f"  ❌ Audio error: {e}")

# Check GPU
print("\nGPU Status:")
try:
    import torch
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("  ⚠ No GPU - CPU mode")
except:
    print("  ❌ PyTorch not available")

print("\n✅ Run this before starting Wallie to check dependencies")
