#!/usr/bin/env python3
"""
Test worker startup to debug initialization failures
"""

import multiprocessing as mp
import time
import queue
from workers.vad_worker import VADWorker
from workers.asr_worker import ASRWorker
from workers.llm_worker import LLMWorker
from workers.tts_worker import TTSWorker

def test_worker_startup():
    """Test individual worker startup and ready signal sending"""
    
    print("Testing Worker Startup and Ready Signals")
    print("=" * 50)
    
    # Create test queues
    queues = {
        'vad_control': mp.Queue(),
        'asr_control': mp.Queue(),
        'llm_control': mp.Queue(),
        'tts_control': mp.Queue(),
        'vad_to_asr': mp.Queue(),
        'asr_to_llm': mp.Queue(),
        'llm_to_tts': mp.Queue()
    }
    
    # Test config
    config = {
        'wake_word': 'wallie',
        'wake_word_sensitivity': 0.7,
        'asr_model': 'tiny.en',
        'asr_device': 'cpu',
        'llm_model': 'microsoft/DialoGPT-small',
        'llm_max_tokens': 100,
        'audio_sample_rate': 16000
    }
    
    # Test each worker
    workers_to_test = [
        ("VAD", VADWorker),
        ("ASR", ASRWorker),
        ("LLM", LLMWorker),
        ("TTS", TTSWorker)
    ]
    
    for worker_name, worker_class in workers_to_test:
        print(f"\nTesting {worker_name} Worker:")
        print("-" * 30)
        
        try:
            # Start worker process
            process = mp.Process(
                target=worker_class.run,
                args=(config, queues),
                name=f"test_{worker_name.lower()}"
            )
            process.start()
            print(f"✓ {worker_name} process started (PID: {process.pid})")
            
            # Wait for ready signal
            control_queue_name = f'{worker_name.lower()}_control'
            control_queue = queues[control_queue_name]
            
            ready_received = False
            start_time = time.time()
            timeout = 15.0  # 15 second timeout
            
            while time.time() - start_time < timeout:
                try:
                    msg = control_queue.get(timeout=0.5)
                    print(f"📨 Received message: {msg}")
                    
                    if msg.get('type') == 'ready' and msg.get('worker') == worker_name.lower():
                        ready_received = True
                        print(f"✅ {worker_name} worker ready signal received!")
                        break
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"❌ Error receiving message: {e}")
                    break
            
            if not ready_received:
                print(f"⚠️  {worker_name} worker did not send ready signal within {timeout}s")
                print(f"   Process alive: {process.is_alive()}")
                if process.is_alive():
                    print("   Worker started but no ready signal - check for silent failures")
                else:
                    print("   Process died - check for startup errors")
            
            # Cleanup
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
            
            # Clear queue
            while True:
                try:
                    control_queue.get_nowait()
                except queue.Empty:
                    break
                    
        except Exception as e:
            print(f"❌ Failed to test {worker_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Set up multiprocessing for Windows
    mp.set_start_method('spawn', force=True)
    test_worker_startup()
