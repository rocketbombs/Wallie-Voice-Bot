#!/usr/bin/env python3
"""
Wallie Voice Bot - Production Startup Script
Optimized configuration for real-time voice interaction
"""

import sys
import os
import time
import signal
import threading
from pathlib import Path
import logging
import multiprocessing as mp

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import main daemon
from wallie_voice_bot import WallieDaemon

class WallieProductionRunner:
    """Production-ready Wallie Voice Bot runner"""
    
    def __init__(self):
        self.daemon = None
        self.daemon_process = None
        self.setup_logging()
        self.setup_signal_handlers()
        
    def setup_logging(self):
        """Setup production logging"""
        log_dir = Path.home() / '.wallie_voice_bot' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"wallie_{time.strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger("wallie.production")
        self.logger.info(f"Log file: {log_file}")
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, 'SIGHUP'):  # Unix only
            signal.signal(signal.SIGHUP, signal_handler)
    
    def get_optimized_config(self):
        """Get production-optimized configuration"""
        return {
            # Wake word configuration
            'wake_word': 'picovoice',  # Use built-in wake word
            'wake_word_sensitivity': 0.7,
            
            # Audio configuration
            'sample_rate': 16000,
            'chunk_size': 1024,
            'audio_device': None,  # Use system default
            'audio_buffer_size': 4096,
            
            # Performance optimization
            'preload_models': True,
            'gpu_memory_fraction': 0.8,
            'cpu_threads': min(4, mp.cpu_count()),
            
            # Latency optimization
            'vad_chunk_ms': 32,  # 32ms chunks for responsive wake word
            'asr_beam_size': 1,  # Faster but less accurate
            'llm_max_tokens': 150,  # Shorter responses
            'tts_speed': 1.1,  # Slightly faster speech
            
            # Model configuration
            'asr_model': 'base',  # Faster than large models
            'llm_model': 'microsoft/DialoGPT-medium',
            'tts_engine': 'pyttsx3',  # Faster than neural TTS
            
            # Cache and storage
            'model_cache_dir': Path.home() / '.wallie_voice_bot' / 'models',
            'conversation_history_size': 10,
            
            # Logging
            'log_level': 'INFO',
            'enable_performance_logging': True,
            
            # Safety and limits
            'max_audio_duration': 30,  # Max 30s speech
            'timeout_speech_end': 2.0,  # 2s silence = end of speech
            'max_response_time': 10.0,  # Max 10s total response time
        }
    
    def check_prerequisites(self):
        """Check system prerequisites"""
        self.logger.info("Checking system prerequisites...")
        
        checks_passed = 0
        total_checks = 6
        
        # Check 1: CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                self.logger.info(f"✓ CUDA GPU: {gpu_name}")
                checks_passed += 1
            else:
                self.logger.warning("⚠️ CUDA not available, using CPU")
        except Exception as e:
            self.logger.error(f"❌ PyTorch check failed: {e}")
        
        # Check 2: Audio system
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            output_devices = [d for d in devices if d['max_output_channels'] > 0]
            
            if input_devices and output_devices:
                self.logger.info(f"✓ Audio: {len(input_devices)} input, {len(output_devices)} output devices")
                checks_passed += 1
            else:
                self.logger.error("❌ Audio devices not available")
        except Exception as e:
            self.logger.error(f"❌ Audio check failed: {e}")
        
        # Check 3: Porcupine access key
        if os.environ.get('PV_ACCESS_KEY'):
            self.logger.info("✓ Porcupine access key available")
            checks_passed += 1
        else:
            self.logger.warning("⚠️ PV_ACCESS_KEY not set, using energy-based VAD")
        
        # Check 4: Critical imports
        try:
            import pvporcupine
            import faster_whisper
            import transformers
            self.logger.info("✓ Critical AI libraries imported")
            checks_passed += 1
        except Exception as e:
            self.logger.error(f"❌ Import check failed: {e}")
        
        # Check 5: Memory check
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            if memory_gb >= 15:
                self.logger.info(f"✓ System memory: {memory_gb:.1f}GB")
                checks_passed += 1
            else:
                self.logger.warning(f"⚠️ Low memory: {memory_gb:.1f}GB (recommended: 16GB)")
        except Exception as e:
            self.logger.error(f"❌ Memory check failed: {e}")
        
        # Check 6: Disk space
        try:
            cache_dir = Path.home() / '.wallie_voice_bot'
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            import shutil
            free_space_gb = shutil.disk_usage(cache_dir).free / (1024**3)
            if free_space_gb >= 5:
                self.logger.info(f"✓ Free disk space: {free_space_gb:.1f}GB")
                checks_passed += 1
            else:
                self.logger.warning(f"⚠️ Low disk space: {free_space_gb:.1f}GB")
        except Exception as e:
            self.logger.error(f"❌ Disk space check failed: {e}")
        
        success_rate = checks_passed / total_checks
        self.logger.info(f"Prerequisites: {checks_passed}/{total_checks} passed ({success_rate:.1%})")
        
        if success_rate >= 0.8:
            self.logger.info("✅ System ready for production")
            return True
        elif success_rate >= 0.6:
            self.logger.warning("⚠️ System has minor issues but should work")
            return True
        else:
            self.logger.error("❌ System has major issues, review configuration")
            return False
    
    def optimize_environment(self):
        """Optimize environment variables for performance"""
        self.logger.info("Optimizing environment for performance...")
        
        # PyTorch optimizations
        os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        
        # OpenMP optimization
        os.environ['OMP_NUM_THREADS'] = str(min(4, mp.cpu_count()))
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        
        # TensorFlow optimizations (if used)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
        
        # Memory optimizations
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        self.logger.info("✓ Environment optimized")
    
    def start_daemon(self):
        """Start the Wallie daemon"""
        self.logger.info("Starting Wallie Voice Bot daemon...")
        
        try:
            config = self.get_optimized_config()
            
            # Create daemon instance
            self.daemon = WallieDaemon(config)
            
            # Start daemon in subprocess for isolation
            def run_daemon():
                try:
                    self.daemon.run()
                except Exception as e:
                    self.logger.error(f"Daemon error: {e}")
                    raise
            
            self.daemon_process = mp.Process(target=run_daemon)
            self.daemon_process.start()
            
            # Monitor daemon startup
            startup_timeout = 30  # 30 seconds for startup
            for i in range(startup_timeout):
                if not self.daemon_process.is_alive():
                    self.logger.error("❌ Daemon process exited during startup")
                    return False
                
                if i % 5 == 0:
                    self.logger.info(f"Daemon starting... ({i}s)")
                
                time.sleep(1)
            
            if self.daemon_process.is_alive():
                self.logger.info("✅ Wallie daemon started successfully")
                return True
            else:
                self.logger.error("❌ Daemon failed to start")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to start daemon: {e}")
            return False
    
    def monitor_daemon(self):
        """Monitor daemon health and performance"""
        self.logger.info("Monitoring daemon (Ctrl+C to stop)...")
        
        start_time = time.time()
        last_health_check = 0
        
        try:
            while True:
                current_time = time.time()
                
                # Check if daemon is still alive
                if not self.daemon_process.is_alive():
                    self.logger.error("❌ Daemon process died unexpectedly")
                    break
                
                # Health check every 30 seconds
                if current_time - last_health_check >= 30:
                    uptime = current_time - start_time
                    self.logger.info(f"✓ Daemon healthy (uptime: {uptime:.0f}s)")
                    last_health_check = current_time
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested by user")
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")
    
    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Initiating graceful shutdown...")
        
        if self.daemon_process and self.daemon_process.is_alive():
            try:
                # Try graceful termination first
                self.daemon_process.terminate()
                self.daemon_process.join(timeout=10)
                
                # Force kill if needed
                if self.daemon_process.is_alive():
                    self.logger.warning("Force killing daemon process")
                    self.daemon_process.kill()
                    self.daemon_process.join()
                
                self.logger.info("✓ Daemon stopped")
                
            except Exception as e:
                self.logger.error(f"Error during shutdown: {e}")
        
        self.logger.info("Shutdown complete")
    
    def run(self):
        """Main production run method"""
        self.logger.info("WALLIE VOICE BOT - PRODUCTION MODE")
        self.logger.info("="*50)
        
        try:
            # Step 1: Check prerequisites
            if not self.check_prerequisites():
                self.logger.error("Prerequisites check failed, aborting")
                return False
            
            # Step 2: Optimize environment
            self.optimize_environment()
            
            # Step 3: Start daemon
            if not self.start_daemon():
                self.logger.error("Failed to start daemon")
                return False
            
            # Step 4: Monitor daemon
            self.monitor_daemon()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Production run failed: {e}")
            return False
        finally:
            self.shutdown()

def main():
    """Main entry point"""
    print("Wallie Voice Bot - Production Runner")
    print("Starting production-optimized voice assistant...")
    print()
    
    # Check if running in interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        response = input("Start Wallie Voice Bot in production mode? (y/n): ").strip().lower()
        if not response.startswith('y'):
            print("Startup cancelled")
            return
    
    # Start production runner
    runner = WallieProductionRunner()
    success = runner.run()
    
    if success:
        print("\n✅ Wallie Voice Bot completed successfully")
    else:
        print("\n❌ Wallie Voice Bot encountered errors")
        sys.exit(1)

if __name__ == "__main__":
    main()
