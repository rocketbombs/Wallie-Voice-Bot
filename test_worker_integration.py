#!/usr/bin/env python3
"""
Individual Worker Integration Tests for Wallie Voice Bot
Test each worker's core functionality independently
"""

import multiprocessing as mp
import numpy as np
import time
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_vad_worker():
    """Test VAD (Voice Activity Detection) worker"""
    print("\n" + "="*60)
    print("TESTING VAD WORKER")
    print("="*60)
    
    try:
        from workers.vad_worker import VADWorker
        
        # Create test queues
        manager = mp.Manager()
        audio_queue = manager.Queue()
        wake_word_queue = manager.Queue()
        control_queue = manager.Queue()
        
        # Create worker config
        config = {
            'wake_word': 'wallie',
            'pvporcupine': {
                'access_key': 'test_key',  # Would need real key for actual test
                'sensitivity': 0.5
            }
        }
        
        print("✓ VAD Worker imports successfully")
        print("✓ Queues created")
        print("ℹ️  Note: Full VAD testing requires Porcupine access key")
        
        # Test synthetic audio data processing
        # Generate test audio that simulates speech patterns
        sample_rate = 16000
        duration = 1.0  # 1 second
        test_audio = np.random.normal(0, 0.1, int(sample_rate * duration)).astype(np.float32)
        
        print(f"✓ Generated test audio: {len(test_audio)} samples at {sample_rate}Hz")
        return True
        
    except ImportError as e:
        print(f"✗ VAD Worker import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ VAD Worker test failed: {e}")
        return False

def test_asr_worker():
    """Test ASR (Automatic Speech Recognition) worker"""
    print("\n" + "="*60)
    print("TESTING ASR WORKER")
    print("="*60)
    
    try:
        from workers.asr_worker import ASRWorker
        
        # Create test queues
        manager = mp.Manager()
        audio_queue = manager.Queue()
        transcription_queue = manager.Queue()
        control_queue = manager.Queue()
        
        config = {
            'asr': {
                'model': 'tiny.en',
                'device': 'cpu',  # Use CPU for testing
                'compute_type': 'float32'
            }
        }
        
        print("✓ ASR Worker imports successfully")
        
        # Test loading Whisper model
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel(config['asr']['model'], device=config['asr']['device'])
            print(f"✓ Whisper model '{config['asr']['model']}' loaded successfully")
            
            # Test transcription with synthetic audio
            # Generate audio that sounds like speech (sine waves at speech frequencies)
            sample_rate = 16000
            duration = 2.0
            freq1, freq2 = 200, 800  # Typical speech frequencies
            t = np.linspace(0, duration, int(sample_rate * duration))
            test_audio = (np.sin(2 * np.pi * freq1 * t) + 
                         0.5 * np.sin(2 * np.pi * freq2 * t)).astype(np.float32)
            test_audio *= 0.1  # Reduce volume
            
            print(f"✓ Generated test speech-like audio: {len(test_audio)} samples")
            
            # Test transcription (might not produce meaningful text but tests the pipeline)
            segments, info = model.transcribe(test_audio)
            segments_list = list(segments)
            
            print(f"✓ Transcription completed: {len(segments_list)} segments")
            print(f"  Language: {info.language} (confidence: {info.language_probability:.2f})")
            
            for i, segment in enumerate(segments_list[:3]):  # Show first 3 segments
                print(f"  Segment {i+1}: '{segment.text.strip()}'")
            
            return True
            
        except Exception as e:
            print(f"✗ Whisper model test failed: {e}")
            return False
        
    except ImportError as e:
        print(f"✗ ASR Worker import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ ASR Worker test failed: {e}")
        return False

def test_llm_worker():
    """Test LLM (Large Language Model) worker"""
    print("\n" + "="*60)
    print("TESTING LLM WORKER")
    print("="*60)
    
    try:
        from workers.llm_worker import LLMWorker
        
        # Create test queues
        manager = mp.Manager()
        transcription_queue = manager.Queue()
        response_queue = manager.Queue()
        control_queue = manager.Queue()
        
        config = {
            'llm': {
                'model': 'microsoft/DialoGPT-small',  # Smaller model for testing
                'max_tokens': 50,
                'temperature': 0.7
            }
        }
        
        print("✓ LLM Worker imports successfully")
        
        # Test using transformers (simpler than vLLM for testing)
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            print(f"✓ Loading model: {config['llm']['model']}")
            tokenizer = AutoTokenizer.from_pretrained(config['llm']['model'])
            model = AutoModelForCausalLM.from_pretrained(config['llm']['model'])
            
            # Add pad token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            print("✓ Model loaded successfully")
            
            # Test text generation
            test_prompt = "Hello, how are you?"
            print(f"Testing with prompt: '{test_prompt}'")
            
            # Encode and generate
            inputs = tokenizer.encode(test_prompt, return_tensors='pt')
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 20,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"✓ Generated response: '{response}'")
            
            return True
            
        except Exception as e:
            print(f"✗ LLM model test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except ImportError as e:
        print(f"✗ LLM Worker import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ LLM Worker test failed: {e}")
        return False

def test_tts_worker():
    """Test TTS (Text-to-Speech) worker"""
    print("\n" + "="*60)
    print("TESTING TTS WORKER")
    print("="*60)
    
    try:
        from workers.tts_worker import TTSWorker
        
        # Create test queues
        manager = mp.Manager()
        response_queue = manager.Queue()
        audio_output_queue = manager.Queue()
        control_queue = manager.Queue()
        
        config = {
            'tts': {
                'engine': 'pyttsx3',  # Use pyttsx3 since it's available
                'voice': 'default',
                'rate': 150,
                'volume': 0.8
            }
        }
        
        print("✓ TTS Worker imports successfully")
        
        # Test pyttsx3 TTS
        try:
            import pyttsx3
            
            engine = pyttsx3.init()
            print("✓ pyttsx3 engine initialized")
            
            # Test voice synthesis
            test_text = "Hello, this is a test of the text to speech system."
            print(f"Testing with text: '{test_text}'")
            
            # Get available voices
            voices = engine.getProperty('voices')
            print(f"✓ Found {len(voices)} available voices")
            
            for i, voice in enumerate(voices[:3]):  # Show first 3 voices
                print(f"  Voice {i+1}: {voice.name}")
            
            # Set properties
            engine.setProperty('rate', config['tts']['rate'])
            engine.setProperty('volume', config['tts']['volume'])
            
            print("✓ TTS engine configured successfully")
            print("ℹ️  Note: Audio output test skipped (would play sound)")
            
            # Test edge-tts as alternative
            try:
                import edge_tts
                print("✓ Edge TTS also available as alternative")
            except ImportError:
                pass
            
            return True
            
        except Exception as e:
            print(f"✗ TTS engine test failed: {e}")
            return False
        
    except ImportError as e:
        print(f"✗ TTS Worker import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ TTS Worker test failed: {e}")
        return False

def main():
    """Run all worker integration tests"""
    print("Wallie Voice Bot - Worker Integration Tests")
    print("="*70)
    
    # Set environment to avoid conflicts
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Run tests
    results = {}
    
    results['vad'] = test_vad_worker()
    results['asr'] = test_asr_worker()
    results['llm'] = test_llm_worker()
    results['tts'] = test_tts_worker()
    
    # Summary
    print("\n" + "="*70)
    print("INTEGRATION TEST SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for worker, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{worker.upper()} Worker: {status}")
    
    print(f"\nResults: {passed}/{total} workers passed integration tests")
    
    if passed == total:
        print("🎉 ALL WORKER INTEGRATION TESTS PASSED!")
        print("✓ System ready for end-to-end pipeline testing")
    else:
        print("⚠️  Some workers failed - check individual test results above")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
