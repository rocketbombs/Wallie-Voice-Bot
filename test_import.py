#!/usr/bin/env python3
"""
Quick test script to verify all imports work
"""

def test_imports():
    """Test all critical imports"""
    print("Testing imports...")
    
    # Test main daemon
    try:
        from wallie_voice_bot import WallieDaemon, WallieConfig
        print("✓ Main daemon import successful")
    except Exception as e:
        print(f"✗ Main daemon import failed: {e}")
        return False
    
    # Test workers
    workers_to_test = [
        ('VAD Worker', 'workers.vad_worker', 'VADWorker'),
        ('ASR Worker', 'workers.asr_worker', 'ASRWorker'), 
        ('LLM Worker', 'workers.llm_worker', 'LLMWorker'),
        ('TTS Worker', 'workers.tts_worker', 'TTSWorker'),
        ('Base Worker', 'workers.worker_base', 'WorkerBase')
    ]
    
    for name, module, class_name in workers_to_test:
        try:
            module_obj = __import__(module, fromlist=[class_name])
            worker_class = getattr(module_obj, class_name)
            print(f"✓ {name} import successful")
        except Exception as e:
            print(f"✗ {name} import failed: {e}")
    
    # Test configuration
    try:
        config = WallieConfig()
        print(f"✓ Configuration created: wake_word='{config.wake_word}'")
    except Exception as e:
        print(f"✗ Configuration creation failed: {e}")
        return False
    
    print("All basic imports successful!")
    return True

if __name__ == "__main__":
    test_imports()
