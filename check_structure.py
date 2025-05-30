#!/usr/bin/env python3
"""
Check and fix Wallie Voice Bot file structure
"""

import os
import shutil
from pathlib import Path

def check_and_fix_structure():
    """Ensure correct file structure"""
    
    # Expected structure
    expected_files = {
        'wallie_voice_bot.py': 'main',
        'workers/__init__.py': 'workers_init',
        'workers/vad_worker.py': 'vad',
        'workers/asr_worker.py': 'asr', 
        'workers/llm_worker.py': 'llm',
        'workers/tts_worker.py': 'tts',
        'requirements.txt': 'deps',
        'setup.py': 'setup',
        'README.md': 'docs'
    }
    
    # Check current directory
    current_dir = Path.cwd()
    print(f"Checking structure in: {current_dir}")
    
    # Create workers directory if missing
    workers_dir = current_dir / 'workers'
    if not workers_dir.exists():
        print("Creating workers/ directory...")
        workers_dir.mkdir()
    
    # Check each file
    missing = []
    for file_path, desc in expected_files.items():
        full_path = current_dir / file_path
        if not full_path.exists():
            missing.append(file_path)
            print(f"❌ Missing: {file_path}")
        else:
            print(f"✅ Found: {file_path}")
    
    # Check for misplaced files
    worker_files = ['vad_worker.py', 'asr_worker.py', 'llm_worker.py', 'tts_worker.py']
    for worker_file in worker_files:
        if (current_dir / worker_file).exists() and not (workers_dir / worker_file).exists():
            print(f"Moving {worker_file} to workers/")
            shutil.move(str(current_dir / worker_file), str(workers_dir / worker_file))
    
    # Check for workers_init.py that should be __init__.py
    if (workers_dir / 'workers_init.py').exists() and not (workers_dir / '__init__.py').exists():
        print("Renaming workers_init.py to __init__.py")
        shutil.move(str(workers_dir / 'workers_init.py'), str(workers_dir / '__init__.py'))
    
    # Create empty __init__.py if missing
    init_file = workers_dir / '__init__.py'
    if not init_file.exists():
        print("Creating workers/__init__.py")
        init_file.write_text('''"""
Wallie Voice Bot Workers
"""

from .vad_worker import VADWorker
from .asr_worker import ASRWorker
from .llm_worker import LLMWorker
from .tts_worker import TTSWorker

__all__ = ['VADWorker', 'ASRWorker', 'LLMWorker', 'TTSWorker']
''')
    
    print("\n" + "="*50)
    if missing:
        print(f"⚠️  Missing files: {len(missing)}")
        print("Please ensure all files from the repository are present")
    else:
        print("✅ File structure is correct!")
    
    return len(missing) == 0

if __name__ == "__main__":
    import sys
    success = check_and_fix_structure()
    sys.exit(0 if success else 1)
