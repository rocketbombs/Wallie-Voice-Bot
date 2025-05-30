#!/usr/bin/env python3
"""
Comprehensive worker functionality tests
Tests each worker component for proper functionality and performance
"""

import asyncio
import time
import threading
import queue
import multiprocessing as mp
import numpy as np
from pathlib import Path
import sys
import json

def test_worker_dependencies():
    """Test if all worker dependencies are available"""
    print("Testing Worker Dependencies")
    print("=" * 50)
    
    dependencies = {
        'VAD Worker': [
            ('pvporcupine', 'Porcupine wake word detection'),
            ('sounddevice', 'Audio capture and playback'),
            ('numpy', 'Audio processing')
        ],
        'ASR Worker': [
            ('faster_whisper', 'Speech recognition'),
            ('torch', 'PyTorch framework'),
        ],
        'LLM Worker': [
            ('vllm', 'vLLM inference engine (optional)'),
            ('torch', 'PyTorch framework'),
            ('transformers', 'Transformers library (fallback)')
        ],
        'TTS Worker': [
            ('TTS', 'Coqui TTS (preferred)'),
            ('edge_tts', 'Edge TTS (fallback)'),
            ('pyttsx3', 'Pyttsx3 (fallback)'),
            ('sounddevice', 'Audio playback')
        ]
    }
    
    results = {}
    
    for worker, deps in dependencies.items():
        print(f"\n{worker}:")
        worker_results = []
        
        for module, description in deps:
            try:
                __import__(module)
                status = "✓ Available"
                available = True
            except ImportError as e:
                status = f"✗ Missing: {e}"
                available = False
            
            print(f"  {module:15} - {status}")
            worker_results.append((module, available, description))
        
        results[worker] = worker_results
    
    return results

def test_audio_system():
    """Test audio input/output capabilities"""
    print("\nTesting Audio System")
    print("=" * 50)
    
    try:
        import sounddevice as sd
        
        # List audio devices
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        output_devices = [d for d in devices if d['max_output_channels'] > 0]
        
        print(f"Input devices: {len(input_devices)} found")
        print(f"Output devices: {len(output_devices)} found")
        
        if input_devices:
            default_input = sd.default.device[0]
            print(f"Default input: {devices[default_input]['name']}")
        
        if output_devices:
            default_output = sd.default.device[1] 
            print(f"Default output: {devices[default_output]['name']}")
        
        # Test basic audio capture (very brief)
        print("\nTesting audio capture (1 second)...")
        duration = 1.0
        sample_rate = 16000
        
        recording = sd.rec(int(duration * sample_rate), 
                          samplerate=sample_rate, 
                          channels=1, dtype='float32')
        sd.wait()
        
        if recording is not None and len(recording) > 0:
            max_level = np.max(np.abs(recording))
            print(f"✓ Audio capture successful (max level: {max_level:.3f})")
            return True
        else:
            print("✗ Audio capture failed")
            return False
            
    except Exception as e:
        print(f"✗ Audio system test failed: {e}")
        return False

def test_gpu_availability():
    """Test GPU and CUDA availability for ML models"""
    print("\nTesting GPU Availability")
    print("=" * 50)
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {'✓' if cuda_available else '✗'}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"GPU devices: {device_count}")
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1024**3
                print(f"  GPU {i}: {props.name} ({memory_gb:.1f}GB)")
            
            # Test basic GPU operation
            x = torch.randn(1000, 1000).cuda()
            y = torch.mm(x, x.t())
            print("✓ Basic GPU computation successful")
            
            return True
        else:
            print("GPU not available, workers will use CPU")
            return False
            
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
        return False

def test_model_loading():
    """Test loading of AI models"""
    print("\nTesting Model Loading")
    print("=" * 50)
    
    results = {}
    
    # Test Whisper model loading
    print("Testing Whisper (ASR)...")
    try:
        from faster_whisper import WhisperModel
        
        # Try tiny model first (fastest to load)
        start_time = time.time()
        model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        load_time = time.time() - start_time
        
        print(f"✓ Whisper tiny.en loaded in {load_time:.1f}s")
        results['whisper'] = True
        
        # Clean up
        del model
        
    except Exception as e:
        print(f"✗ Whisper loading failed: {e}")
        results['whisper'] = False
    
    # Test TTS model availability
    print("\nTesting TTS engines...")
    tts_engines = []
    
    # Test Coqui TTS
    try:
        from TTS.api import TTS
        print("✓ Coqui TTS available")
        tts_engines.append('coqui')
    except ImportError:
        print("✗ Coqui TTS not available")
    
    # Test Edge TTS
    try:
        import edge_tts
        print("✓ Edge TTS available")
        tts_engines.append('edge')
    except ImportError:
        print("✗ Edge TTS not available")
    
    # Test pyttsx3
    try:
        import pyttsx3
        print("✓ pyttsx3 available")
        tts_engines.append('pyttsx3')
    except ImportError:
        print("✗ pyttsx3 not available")
    
    results['tts_engines'] = tts_engines
    
    # Test LLM availability
    print("\nTesting LLM engines...")
    
    # Test vLLM
    try:
        from vllm import LLM
        print("✓ vLLM available")
        results['vllm'] = True
    except ImportError:
        print("✗ vLLM not available")
        results['vllm'] = False
    
    # Test transformers (fallback)
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("✓ Transformers available")
        results['transformers'] = True
    except ImportError:
        print("✗ Transformers not available")
        results['transformers'] = False
    
    return results

def test_worker_initialization():
    """Test if workers can be initialized properly"""
    print("\nTesting Worker Initialization")
    print("=" * 50)
    
    # Mock configuration
    config = {
        'wake_word': 'wallie',
        'wake_word_sensitivity': 0.7,
        'asr_model': 'tiny.en',
        'asr_device': 'cpu',
        'asr_compute_type': 'int8',
        'llm_model': 'microsoft/DialoGPT-small',  # Small test model
        'llm_max_tokens': 50,
        'llm_temperature': 0.7,
        'llm_gpu_memory_fraction': 0.3,
        'tts_model': 'tts_models/en/ljspeech/tacotron2-DDC',
        'audio_sample_rate': 16000,
        'audio_chunk_size': 512
    }
    
    # Create mock queues
    queues = {
        'vad_to_asr': mp.Queue(),
        'asr_to_llm': mp.Queue(),
        'llm_to_tts': mp.Queue(),
        'vad_control': mp.Queue(),
        'asr_control': mp.Queue(),
        'llm_control': mp.Queue(),
        'tts_control': mp.Queue()
    }
    
    results = {}
    
    # Test VAD Worker
    print("Testing VAD Worker initialization...")
    try:
        from workers.vad_worker import VADWorker
        vad_worker = VADWorker(config, queues)
        print("✓ VAD Worker initialized")
        results['vad'] = True
    except Exception as e:
        print(f"✗ VAD Worker failed: {e}")
        results['vad'] = False
    
    # Test ASR Worker
    print("Testing ASR Worker initialization...")
    try:
        from workers.asr_worker import ASRWorker
        asr_worker = ASRWorker(config, queues)
        print("✓ ASR Worker initialized")
        results['asr'] = True
    except Exception as e:
        print(f"✗ ASR Worker failed: {e}")
        results['asr'] = False
    
    # Test LLM Worker
    print("Testing LLM Worker initialization...")
    try:
        from workers.llm_worker import LLMWorker
        llm_worker = LLMWorker(config, queues)
        print("✓ LLM Worker initialized")
        results['llm'] = True
    except Exception as e:
        print(f"✗ LLM Worker failed: {e}")
        results['llm'] = False
    
    # Test TTS Worker
    print("Testing TTS Worker initialization...")
    try:
        from workers.tts_worker import TTSWorker
        tts_worker = TTSWorker(config, queues)
        print("✓ TTS Worker initialized")
        results['tts'] = True
    except Exception as e:
        print(f"✗ TTS Worker failed: {e}")
        results['tts'] = False
    
    return results

def test_queue_communication():
    """Test inter-process queue communication"""
    print("\nTesting Queue Communication")
    print("=" * 50)
    
    try:
        # Set environment variable to avoid OpenMP conflicts
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        
        # Test basic queue operations
        test_queue = mp.Queue()
        
        # Test putting and getting messages (smaller data to avoid serialization issues)
        test_messages = [
            {'type': 'wake_word_detected', 'timestamp': time.time()},
            {'type': 'audio_chunk', 'data': [0.1, 0.2, 0.3]},  # Small test data
            {'type': 'transcription', 'text': 'hello world'},
            {'type': 'response', 'text': 'Hello! How can I help you?'},
            {'type': 'audio_generated', 'audio_data': [0.4, 0.5, 0.6]}
        ]
        
        # Put messages
        for i, msg in enumerate(test_messages):
            test_queue.put(msg)
        
        # Get messages with timeout (more reliable than empty() check)
        received = []
        for i in range(len(test_messages)):
            try:
                msg = test_queue.get(timeout=1)
                received.append(msg)
            except:
                break  # Timeout or queue empty
        
        if len(received) == len(test_messages):
            print("✓ Queue communication working")
            return True
        else:
            print(f"✗ Queue communication failed: sent {len(test_messages)}, received {len(received)}")
            # Show what was received for debugging
            for i, msg in enumerate(received):
                print(f"  Received {i+1}: {msg['type']}")
            return False
            
    except Exception as e:
        print(f"✗ Queue test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all worker tests"""
    print("Wallie Voice Bot - Worker Functionality Tests")
    print("=" * 60)
    
    test_results = {}
    
    # Run all tests
    test_results['dependencies'] = test_worker_dependencies()
    test_results['audio'] = test_audio_system()
    test_results['gpu'] = test_gpu_availability()
    test_results['models'] = test_model_loading()
    test_results['workers'] = test_worker_initialization()
    test_results['queues'] = test_queue_communication()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    # Dependency summary
    print("\nDependency Status:")
    for worker, deps in test_results['dependencies'].items():
        available_count = sum(1 for _, available, _ in deps if available)
        total_count = len(deps)
        print(f"  {worker}: {available_count}/{total_count} dependencies available")
    
    # Core functionality
    core_tests = ['audio', 'gpu', 'queues']
    core_passed = sum(1 for test in core_tests if test_results.get(test, False))
    print(f"\nCore System Tests: {core_passed}/{len(core_tests)} passed")
    
    # Worker initialization
    if 'workers' in test_results:
        worker_tests = test_results['workers']
        workers_passed = sum(1 for result in worker_tests.values() if result)
        print(f"Worker Initialization: {workers_passed}/{len(worker_tests)} workers successful")
    
    # Model loading
    if 'models' in test_results:
        models = test_results['models']
        model_summary = []
        if models.get('whisper'):
            model_summary.append("ASR✓")
        if models.get('tts_engines'):
            model_summary.append(f"TTS✓({len(models['tts_engines'])} engines)")
        if models.get('vllm') or models.get('transformers'):
            model_summary.append("LLM✓")
        
        print(f"AI Models: {', '.join(model_summary) if model_summary else 'None available'}")
    
    # Overall status
    critical_tests = ['audio', 'queues']
    critical_passed = all(test_results.get(test, False) for test in critical_tests)
    
    if critical_passed:
        print("\n✓ SYSTEM READY for worker integration testing")
    else:
        print("\n✗ SYSTEM NOT READY - Critical tests failed")
    
    return test_results

if __name__ == "__main__":
    results = main()
    
    # Exit with appropriate code
    critical_tests = ['audio', 'queues']
    critical_passed = all(results.get(test, False) for test in critical_tests)
    sys.exit(0 if critical_passed else 1)
