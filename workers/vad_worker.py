"""
VAD Worker - Wake word detection using Porcupine
Continuously monitors audio for "wallie" trigger
"""

import time
import numpy as np
import sounddevice as sd
from typing import Optional, Dict, Any
import multiprocessing as mp
import logging
import json
from pathlib import Path
import struct
import threading
import queue
import os

# Try to import pvporcupine
PORCUPINE_AVAILABLE = False
try:
    import pvporcupine
    PORCUPINE_AVAILABLE = True
except ImportError:
    pass

class VADWorker:
    """Voice Activity Detection with Porcupine wake word"""
    
    @staticmethod
    def run(config: Dict[str, Any], queues: Dict[str, mp.Queue]):
        """Main worker process entry point"""
        worker = VADWorker(config, queues)
        worker.start()
    
    def __init__(self, config: Dict[str, Any], queues: Dict[str, mp.Queue]):
        self.config = config
        self.queues = queues
        self.logger = self._setup_logging()
        
        # Audio configuration
        self.sample_rate = 16000  # Porcupine requires 16kHz
        self.frame_length = 512   # Porcupine frame size
        self.channels = 1
        
        # Wake word detection
        self.porcupine: Optional[pvporcupine.Porcupine] = None
        self.wake_word = config.get('wake_word', 'wallie')
        self.sensitivity = config.get('wake_word_sensitivity', 0.7)
        
        # Audio buffer for post-wake-word capture
        self.audio_buffer = []
        self.buffer_duration = 0.5  # Pre-roll audio in seconds
        self.buffer_size = int(self.sample_rate * self.buffer_duration)
        
        # State management
        self.listening_for_speech = False
        self.speech_start_time = None
        self.silence_threshold = 0.01
        self.silence_duration = 1.5  # End of speech after 1.5s silence
        self.last_speech_time = None
        
        # Performance tracking
        self.wake_word_count = 0
        self.audio_overruns = 0
        
        # Audio processing queue
        self.audio_queue = queue.Queue()
        self.running = True
        
        # Use dedicated control queue
        self.control_queue = queues['vad_control']
        
    def _setup_logging(self) -> logging.Logger:
        """Setup worker-specific logging"""
        logger = logging.getLogger("wallie.vad")
        logger.setLevel(logging.INFO)
        return logger
    
    def initialize_porcupine(self):
        """Initialize Porcupine wake word engine"""
        if not PORCUPINE_AVAILABLE:
            self.logger.warning("Porcupine not available, using energy-based VAD")
            return
            
        try:
            # Check for API key
            if not os.environ.get('PV_ACCESS_KEY'):
                self.logger.warning("PV_ACCESS_KEY environment variable not set! Using energy-based VAD")
                return
            
            # Try custom wake word first
            wake_word_path = Path.home() / ".wallie_voice_bot" / "wake_words" / f"{self.wake_word}.ppn"
            
            if wake_word_path.exists():
                self.logger.info(f"Loading custom wake word: {wake_word_path}")
                self.porcupine = pvporcupine.create(
                    access_key=os.environ['PV_ACCESS_KEY'],
                    keyword_paths=[str(wake_word_path)],
                    sensitivities=[self.sensitivity]
                )
            else:
                # Use built-in keywords
                self.logger.info(f"Using built-in wake word: {self.wake_word}")
                keywords = ["picovoice", "bumblebee", "alexa", "ok google", "hey google", "hey siri", "jarvis", "computer"]
                
                if self.wake_word.lower() in keywords:
                    self.porcupine = pvporcupine.create(
                        access_key=os.environ['PV_ACCESS_KEY'],
                        keywords=[self.wake_word.lower()],
                        sensitivities=[self.sensitivity]
                    )
                else:
                    # Default to "picovoice"
                    self.logger.warning(f"Wake word '{self.wake_word}' not found, using 'picovoice'")
                    self.porcupine = pvporcupine.create(
                        access_key=os.environ['PV_ACCESS_KEY'],
                        keywords=["picovoice"],
                        sensitivities=[self.sensitivity]
                    )
            
            self.logger.info(f"Porcupine initialized (frame_length={self.porcupine.frame_length})")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Porcupine: {e}")
            self.porcupine = None
    
    def audio_callback(self, indata: np.ndarray, frames: int, time_info: Dict, status: sd.CallbackFlags):
        """Process incoming audio frames"""
        if status:
            self.audio_overruns += 1
            if self.audio_overruns % 100 == 0:
                self.logger.warning(f"Audio overruns: {self.audio_overruns}")
        
        # Convert to int16 for Porcupine
        audio_int16 = (indata[:, 0] * 32767).astype(np.int16)
        
        # Add to rolling buffer
        self.audio_buffer.extend(audio_int16)
        if len(self.audio_buffer) > self.buffer_size:
            self.audio_buffer = self.audio_buffer[-self.buffer_size:]
        
        # Process in Porcupine frame chunks
        for i in range(0, len(audio_int16) - self.frame_length + 1, self.frame_length):
            frame = audio_int16[i:i + self.frame_length]
            
            if not self.listening_for_speech:
                # Check for wake word
                if self.porcupine:
                    try:
                        keyword_index = self.porcupine.process(frame)
                        if keyword_index >= 0:
                            self.handle_wake_word_detected()
                    except Exception as e:
                        self.logger.error(f"Porcupine process error: {e}")
                else:
                    # Energy-based trigger
                    energy = calculate_energy(frame)
                    if energy > 0.1:  # Higher threshold for wake detection
                        self.handle_wake_word_detected()
            else:
                # Send audio to ASR
                self.process_speech_audio(frame)
    
    def handle_wake_word_detected(self):
        """Handle wake word detection"""
        self.wake_word_count += 1
        self.logger.info(f"Wake word detected (#{self.wake_word_count})")
        
        # Send interrupt signal through control queue
        try:
            self.control_queue.put_nowait({
                'type': 'wake_word_detected',
                'timestamp': time.time(),
                'count': self.wake_word_count
            })
        except:
            pass
        
        # Start listening for speech
        self.listening_for_speech = True
        self.speech_start_time = time.time()
        self.last_speech_time = time.time()
        
        # Send buffered audio (pre-roll) to ASR
        if self.audio_buffer:
            self.queues['vad_to_asr'].put({
                'type': 'audio_chunk',
                'data': np.array(self.audio_buffer[-self.frame_length*4:], dtype=np.int16),
                'timestamp': time.time(),
                'is_first': True
            })
    
    def process_speech_audio(self, frame: np.ndarray):
        """Process audio during speech recognition"""
        # Simple energy-based speech detection
        energy = calculate_energy(frame)
        
        if energy > self.silence_threshold:
            self.last_speech_time = time.time()
        
        # Send audio chunk to ASR
        try:
            self.queues['vad_to_asr'].put_nowait({
                'type': 'audio_chunk',
                'data': frame,
                'timestamp': time.time(),
                'is_first': False
            })
        except:
            self.logger.warning("ASR queue full, dropping audio")
        
        # Check for end of speech
        if time.time() - self.last_speech_time > self.silence_duration:
            self.handle_end_of_speech()
    
    def handle_end_of_speech(self):
        """Handle end of speech detection"""
        duration = time.time() - self.speech_start_time
        self.logger.info(f"End of speech detected (duration={duration:.1f}s)")
        
        # Send end marker to ASR
        try:
            self.queues['vad_to_asr'].put({
                'type': 'end_of_speech',
                'timestamp': time.time(),
                'duration': duration
            })
        except:
            pass
        
        # Reset state
        self.listening_for_speech = False
        self.speech_start_time = None
        self.last_speech_time = None
    
    def handle_control_message(self, msg: Dict[str, Any]):
        """Handle control messages from main daemon"""
        msg_type = msg.get('type')
        
        if msg_type == 'abort':
            # Reset state
            self.listening_for_speech = False
            self.speech_start_time = None
            self.last_speech_time = None
            self.logger.info("VAD aborted")
            
        elif msg_type == 'config_reload':
            new_config = msg.get('config', {})
            self.sensitivity = new_config.get('wake_word_sensitivity', self.sensitivity)
            
            # Recreate Porcupine if wake word changed
            new_wake_word = new_config.get('wake_word', self.wake_word)
            if new_wake_word != self.wake_word:
                self.wake_word = new_wake_word
                if self.porcupine:
                    self.porcupine.delete()
                self.initialize_porcupine()
    
    def start(self):
        """Main worker loop"""
        try:
            # Initialize Porcupine
            self.initialize_porcupine()
            
            # Report ready
            self.control_queue.put({
                'type': 'ready',
                'worker': 'vad',
                'timestamp': time.time()
            })
            
            # Start audio stream
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32',
                blocksize=self.frame_length,
                callback=self.audio_callback
            ):
                self.logger.info("VAD worker started, listening for wake word...")
                
                last_metrics_time = time.time()
                
                # Main loop - handle control messages
                while self.running:
                    try:
                        # Non-blocking check for control messages
                        msg = self.control_queue.get(timeout=0.1)
                        self.handle_control_message(msg)
                    except:
                        pass
                    
                    # Log periodic metrics
                    if time.time() - last_metrics_time >= 30:
                        self.logger.info("VAD metrics", extra={
                            "stage": "vad",
                            "wake_word_count": self.wake_word_count,
                            "audio_overruns": self.audio_overruns,
                            "using_porcupine": self.porcupine is not None
                        })
                        last_metrics_time = time.time()
                    
                    time.sleep(0.01)
                    
        except KeyboardInterrupt:
            self.logger.info("VAD worker interrupted")
        except Exception as e:
            self.logger.error(f"VAD worker error: {e}", exc_info=True)
        finally:
            if self.porcupine:
                self.porcupine.delete()
            self.running = False
            self.logger.info("VAD worker stopped")

# Energy-based VAD utilities
def calculate_energy(frame: np.ndarray) -> float:
    """Calculate frame energy for VAD"""
    return np.sqrt(np.mean(frame.astype(np.float32) ** 2))

def adaptive_threshold(energies: list, percentile: float = 30) -> float:
    """Calculate adaptive silence threshold"""
    if len(energies) < 10:
        return 0.01
    return np.percentile(energies, percentile)