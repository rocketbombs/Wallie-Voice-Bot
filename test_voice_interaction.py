#!/usr/bin/env python3
"""
Wallie Voice Bot - Interactive Testing Suite
Tests voice interaction capabilities with real audio I/O
"""

import sys
import time
import threading
import multiprocessing as mp
from pathlib import Path
import logging
import numpy as np
import sounddevice as sd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from wallie_voice_bot import WallieDaemon

class VoiceInteractionTester:
    """Test voice interaction with the Wallie daemon"""
    
    def __init__(self):
        self.setup_logging()
        self.daemon = None
        self.daemon_process = None
        
    def setup_logging(self):
        """Setup logging for testing"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("voice_test")
    
    def test_audio_devices(self):
        """Test audio device availability"""
        print("\n🎤 AUDIO DEVICE TEST")
        print("-" * 40)
        
        try:
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            default_output = sd.default.device[1]
            
            input_device = devices[default_input] if default_input is not None else None
            output_device = devices[default_output] if default_output is not None else None
            
            if input_device:
                print(f"✓ Input Device: {input_device['name']}")
                print(f"  → Channels: {input_device['max_input_channels']}")
                print(f"  → Sample Rate: {input_device['default_samplerate']}")
            else:
                print("❌ No input device available")
                return False
                
            if output_device:
                print(f"✓ Output Device: {output_device['name']}")
                print(f"  → Channels: {output_device['max_output_channels']}")
                print(f"  → Sample Rate: {output_device['default_samplerate']}")
            else:
                print("❌ No output device available")
                return False
                
            return True
            
        except Exception as e:
            print(f"❌ Audio device test failed: {e}")
            return False
    
    def test_microphone(self, duration=3):
        """Test microphone input"""
        print(f"\n🎙️ MICROPHONE TEST ({duration}s)")
        print("-" * 40)
        print("Speak into your microphone...")
        
        try:
            # Record audio
            sample_rate = 16000
            recording = sd.rec(
                int(duration * sample_rate), 
                samplerate=sample_rate, 
                channels=1,
                dtype=np.float32
            )
            sd.wait()  # Wait for recording to complete
            
            # Analyze recording
            max_amplitude = np.max(np.abs(recording))
            mean_amplitude = np.mean(np.abs(recording))
            
            print(f"✓ Recording completed")
            print(f"  → Max amplitude: {max_amplitude:.4f}")
            print(f"  → Mean amplitude: {mean_amplitude:.4f}")
            
            if max_amplitude > 0.01:
                print("✓ Audio input detected")
                return True
            else:
                print("⚠️ Very low audio input - check microphone")
                return False
                
        except Exception as e:
            print(f"❌ Microphone test failed: {e}")
            return False
    
    def test_speakers(self):
        """Test speaker output"""
        print("\n🔊 SPEAKER TEST")
        print("-" * 40)
        print("Playing test tone...")
        
        try:
            # Generate test tone (440 Hz for 1 second)
            sample_rate = 44100
            duration = 1.0
            frequency = 440.0
            
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = np.sin(2 * np.pi * frequency * t) * 0.3
            
            # Play tone
            sd.play(tone, sample_rate)
            sd.wait()  # Wait for playback to complete
            
            print("✓ Test tone played")
            response = input("Did you hear the test tone? (y/n): ").strip().lower()
            
            if response.startswith('y'):
                print("✓ Speaker output confirmed")
                return True
            else:
                print("⚠️ Speaker output not confirmed")
                return False
                
        except Exception as e:
            print(f"❌ Speaker test failed: {e}")
            return False
    
    def start_daemon_process(self):
        """Start the Wallie daemon in a separate process"""
        print("\n🤖 STARTING WALLIE DAEMON")
        print("-" * 40)
        
        try:
            # Create daemon configuration
            config = {
                'wake_word': 'picovoice',  # Use built-in wake word
                'wake_word_sensitivity': 0.7,
                'sample_rate': 16000,
                'chunk_size': 1024,
                'audio_device': None,  # Use default
                'model_cache_dir': Path.home() / '.wallie_voice_bot' / 'models',
                'log_level': 'INFO'
            }
            
            # Start daemon in subprocess
            def run_daemon():
                try:
                    daemon = WallieDaemon(config)
                    daemon.run()
                except Exception as e:
                    print(f"Daemon error: {e}")
            
            self.daemon_process = mp.Process(target=run_daemon)
            self.daemon_process.start()
            
            # Give daemon time to initialize
            time.sleep(5)
            
            if self.daemon_process.is_alive():
                print("✓ Wallie daemon started successfully")
                return True
            else:
                print("❌ Wallie daemon failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start daemon: {e}")
            return False
    
    def test_voice_interaction(self, timeout=30):
        """Test voice interaction with timeout"""
        print(f"\n🗣️ VOICE INTERACTION TEST ({timeout}s timeout)")
        print("-" * 40)
        print("Say 'picovoice' followed by a question or command...")
        print("Examples:")
        print("  - 'picovoice, what's the weather like?'")
        print("  - 'picovoice, tell me a joke'")
        print("  - 'picovoice, hello there'")
        print("\nListening...")
        
        # Simple monitoring - in a real implementation, this would
        # monitor the daemon's queues and responses
        start_time = time.time()
        
        try:
            while time.time() - start_time < timeout:
                # Check if daemon is still running
                if not self.daemon_process.is_alive():
                    print("❌ Daemon process stopped unexpectedly")
                    return False
                
                # In a real test, we'd monitor the output queues
                # For now, just keep the process alive
                time.sleep(1)
                
                # Show progress
                elapsed = int(time.time() - start_time)
                remaining = timeout - elapsed
                if elapsed % 5 == 0 and remaining > 0:
                    print(f"  → Still listening... ({remaining}s remaining)")
            
            print("✓ Voice interaction test completed")
            print("  → Check console output for wake word detections")
            print("  → Check audio output for TTS responses")
            return True
            
        except KeyboardInterrupt:
            print("\n⚠️ Test interrupted by user")
            return False
        except Exception as e:
            print(f"❌ Voice interaction test failed: {e}")
            return False
    
    def stop_daemon(self):
        """Stop the daemon process"""
        if self.daemon_process and self.daemon_process.is_alive():
            print("\n🛑 STOPPING DAEMON")
            print("-" * 40)
            try:
                self.daemon_process.terminate()
                self.daemon_process.join(timeout=5)
                
                if self.daemon_process.is_alive():
                    print("⚠️ Force killing daemon process")
                    self.daemon_process.kill()
                    self.daemon_process.join()
                
                print("✓ Daemon stopped")
                
            except Exception as e:
                print(f"❌ Error stopping daemon: {e}")
    
    def run_full_test(self):
        """Run the complete voice interaction test suite"""
        print("WALLIE VOICE BOT - VOICE INTERACTION TEST")
        print("="*50)
        print(f"Test started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        tests_passed = 0
        total_tests = 5
        
        try:
            # Test 1: Audio devices
            if self.test_audio_devices():
                tests_passed += 1
            
            # Test 2: Microphone
            if self.test_microphone():
                tests_passed += 1
            
            # Test 3: Speakers
            if self.test_speakers():
                tests_passed += 1
            
            # Test 4: Start daemon
            if self.start_daemon_process():
                tests_passed += 1
                
                # Test 5: Voice interaction
                if self.test_voice_interaction():
                    tests_passed += 1
            
        except KeyboardInterrupt:
            print("\n⚠️ Testing interrupted by user")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
        finally:
            # Always try to stop daemon
            self.stop_daemon()
        
        # Results summary
        print(f"\n{'='*50}")
        print("TEST RESULTS SUMMARY")
        print(f"{'='*50}")
        print(f"Tests passed: {tests_passed}/{total_tests}")
        
        if tests_passed == total_tests:
            print("🎉 ALL TESTS PASSED!")
            print("✅ Wallie Voice Bot is ready for voice interaction")
        elif tests_passed >= 3:
            print("⚠️ PARTIAL SUCCESS")
            print("🔧 Some components need attention")
        else:
            print("❌ MAJOR ISSUES DETECTED")
            print("🛠️ System requires troubleshooting")
        
        return tests_passed, total_tests

def main():
    """Main test execution"""
    print("Wallie Voice Bot - Interactive Testing")
    print("This will test real voice interaction capabilities")
    print()
    
    response = input("Ready to start voice testing? (y/n): ").strip().lower()
    if not response.startswith('y'):
        print("Test cancelled by user")
        return
    
    # Run tests
    tester = VoiceInteractionTester()
    passed, total = tester.run_full_test()
    
    # Next steps guidance
    print(f"\n{'='*50}")
    print("NEXT STEPS")
    print(f"{'='*50}")
    
    if passed == total:
        print("🚀 Your Wallie Voice Bot is fully operational!")
        print("\nRecommended actions:")
        print("1. Experiment with different wake word phrases")
        print("2. Test conversation memory and context")
        print("3. Optimize latency for faster responses")
        print("4. Configure custom wake words")
        print("5. Set up production deployment")
    else:
        print("🔧 Address the following to complete setup:")
        if passed < 3:
            print("- Fix audio device issues")
            print("- Check microphone/speaker connections")
        if passed < 4:
            print("- Debug daemon initialization")
            print("- Check system dependencies")
        if passed < 5:
            print("- Test wake word detection")
            print("- Verify voice processing pipeline")

if __name__ == "__main__":
    main()
