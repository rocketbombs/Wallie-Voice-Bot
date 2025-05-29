"""
TTS Worker - Text-to-Speech using multiple engines
Target: ≤30ms to first audio chunk
"""

import time
import numpy as np
from typing import Optional, Dict, Any, List, Union
import multiprocessing as mp
import logging
import torch
import sounddevice as sd
import threading
import queue
from pathlib import Path
import io
import asyncio

# Try multiple TTS engines
TTS_AVAILABLE = False
EDGE_TTS_AVAILABLE = False
PYTTSX3_AVAILABLE = False

try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    pass

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    pass

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    pass

class TTSWorker:
    """Streaming TTS with multiple engine support"""
    
    @staticmethod
    def run(config: Dict[str, Any], queues: Dict[str, mp.Queue]):
        """Main worker process entry point"""
        worker = TTSWorker(config, queues)
        worker.start()
    
    def __init__(self, config: Dict[str, Any], queues: Dict[str, mp.Queue]):
        self.config = config
        self.queues = queues
        self.logger = self._setup_logging()
        
        # TTS configuration
        self.tts_engine = config.get('tts_engine', 'auto')  # auto, coqui, edge, pyttsx3
        self.speaker_wav = config.get('tts_speaker_wav')
        self.language = config.get('tts_language', 'en')
        self.voice_name = config.get('tts_voice', 'en-US-AriaNeural')  # For edge-tts
        
        # Audio configuration
        self.sample_rate = 24000
        self.output_sample_rate = 48000
        self.chunk_size = 1024
        
        # TTS engines
        self.tts = None
        self.tts_engine_name = None
        self.pyttsx_engine = None
        
        # Audio playback
        self.audio_queue = queue.Queue(maxsize=100)
        self.playback_thread: Optional[threading.Thread] = None
        self.is_playing = False
        
        # Performance tracking
        self.synthesis_count = 0
        self.first_chunk_latencies = []
        self.total_audio_duration = 0.0
        
        # Use dedicated control queue
        self.control_queue = queues['tts_control']
        self.running = True
        
    def _setup_logging(self) -> logging.Logger:
        """Setup worker-specific logging"""
        logger = logging.getLogger("wallie.tts")
        logger.setLevel(logging.INFO)
        return logger
    
    def initialize_tts(self):
        """Initialize TTS with fallback options"""
        # Auto-select best available engine
        if self.tts_engine == 'auto':
            if PYTTSX3_AVAILABLE:
                self.tts_engine = 'pyttsx3'
            elif EDGE_TTS_AVAILABLE:
                self.tts_engine = 'edge'
            elif TTS_AVAILABLE and torch.cuda.is_available():
                self.tts_engine = 'coqui'
            else:
                self.logger.error("No TTS engine available!")
                # Use a simple fallback
                self.tts_engine = 'dummy'
        
        # Initialize selected engine
        if self.tts_engine == 'edge' and EDGE_TTS_AVAILABLE:
            self.logger.info("Using Edge TTS (online)")
            self.tts_engine_name = 'edge'
            # Edge TTS is initialized per-request
            
        elif self.tts_engine == 'pyttsx3' and PYTTSX3_AVAILABLE:
            self.logger.info("Using pyttsx3 (offline)")
            self.tts_engine_name = 'pyttsx3'
            self.pyttsx_engine = pyttsx3.init()
            
            # Configure voice
            voices = self.pyttsx_engine.getProperty('voices')
            if voices:
                # Try to find English voice
                for voice in voices:
                    if 'english' in voice.name.lower():
                        self.pyttsx_engine.setProperty('voice', voice.id)
                        break
            
            # Set properties
            self.pyttsx_engine.setProperty('rate', 180)  # Speed
            self.pyttsx_engine.setProperty('volume', 0.9)
            
        elif self.tts_engine == 'coqui' and TTS_AVAILABLE:
            self.logger.info("Using Coqui TTS (offline)")
            self.tts_engine_name = 'coqui'
            try:
                self.tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=torch.cuda.is_available())
            except:
                self.logger.warning("Failed to load Coqui model, trying pyttsx3")
                self.tts_engine = 'pyttsx3'
                self.initialize_tts()
                return
        
        elif self.tts_engine == 'dummy':
            self.logger.warning("Using dummy TTS (no audio output)")
            self.tts_engine_name = 'dummy'
        
        else:
            self.logger.error(f"TTS engine '{self.tts_engine}' not available, using dummy")
            self.tts_engine = 'dummy'
            self.tts_engine_name = 'dummy'
        
        # Warm up
        if self.tts_engine_name != 'dummy':
            self.logger.info(f"Warming up {self.tts_engine_name}...")
            self._synthesize_text("Hello, I am ready.", streaming=False)
        
        self.logger.info("TTS ready")
    
    def _synthesize_text(self, text: str, streaming: bool = True) -> Optional[np.ndarray]:
        """Synthesize speech using available engine"""
        try:
            start_time = time.perf_counter()
            
            if self.tts_engine_name == 'edge':
                # Edge TTS (async)
                audio = asyncio.run(self._synthesize_edge(text))
                
            elif self.tts_engine_name == 'pyttsx3':
                # PyTTSX3
                audio = self._synthesize_pyttsx(text)
                
            elif self.tts_engine_name == 'coqui':
                # Coqui TTS
                audio = self._synthesize_coqui(text)
                
            elif self.tts_engine_name == 'dummy':
                # Dummy audio
                audio = self._synthesize_dummy(text)
            
            else:
                return None
            
            # Track latency
            if audio is not None and len(audio) > 0:
                latency = (time.perf_counter() - start_time) * 1000
                if streaming:
                    self.first_chunk_latencies.append(latency)
                    self.logger.info(f"First audio chunk in {latency:.1f}ms")
            
            return audio
            
        except Exception as e:
            self.logger.error(f"TTS synthesis error: {e}")
            return None
    
    async def _synthesize_edge(self, text: str) -> Optional[np.ndarray]:
        """Synthesize using Edge TTS"""
        # Create communication object
        tts = edge_tts.Communicate(text, self.voice_name)
        
        # Generate audio
        audio_data = bytearray()
        async for chunk in tts.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
        
        # Convert to numpy
        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        return audio
    
    def _synthesize_pyttsx(self, text: str) -> Optional[np.ndarray]:
        """Synthesize using pyttsx3"""
        # Save to temporary buffer
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Generate audio file
            self.pyttsx_engine.save_to_file(text, tmp_path)
            self.pyttsx_engine.runAndWait()
            
            # Load audio
            import wave
            with wave.open(tmp_path, 'rb') as wav:
                frames = wav.readframes(wav.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                self.sample_rate = wav.getframerate()
            
            return audio
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def _synthesize_coqui(self, text: str) -> Optional[np.ndarray]:
        """Synthesize using Coqui TTS"""
        wav = self.tts.tts(text)
        
        if isinstance(wav, list):
            wav = np.array(wav, dtype=np.float32)
        elif torch.is_tensor(wav):
            wav = wav.cpu().numpy()
        
        return wav
    
    def _synthesize_dummy(self, text: str) -> Optional[np.ndarray]:
        """Generate dummy audio for testing"""
        # Generate silence of appropriate length
        duration = len(text) * 0.05  # ~50ms per character
        samples = int(duration * self.output_sample_rate)
        return np.zeros(samples, dtype=np.float32)
    
    def audio_playback_thread(self):
        """Background thread for audio playback"""
        try:
            # Simple playback using sounddevice
            self.logger.info("Audio playback started")
            
            while self.is_playing:
                try:
                    # Get audio chunk from queue
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    
                    if audio_chunk is None:  # Stop signal
                        break
                    
                    # Play audio directly (skip for dummy engine)
                    if self.tts_engine_name != 'dummy':
                        sd.play(audio_chunk, self.output_sample_rate)
                        sd.wait()  # Wait for playback to finish
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Playback error: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to start audio playback: {e}")
        finally:
            self.logger.info("Audio playback stopped")
    
    def process_llm_response(self, response: Dict[str, Any]):
        """Process response from LLM and synthesize speech"""
        text = response.get('text', '')
        request_id = response.get('request_id')
        
        if not text:
            return
        
        self.logger.info(f"Synthesizing: {text[:50]}...")
        
        # Split text into sentences for streaming
        sentences = self._split_sentences(text)
        
        # Start playback if not already playing
        if not self.is_playing:
            self.is_playing = True
            self.playback_thread = threading.Thread(target=self.audio_playback_thread)
            self.playback_thread.start()
        
        # Synthesize each sentence
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            # Check for abort signal
            try:
                abort_msg = self.control_queue.get_nowait()
                if abort_msg.get('type') == 'abort':
                    self.logger.info("TTS aborted")
                    self.stop_playback()
                    return
            except:
                pass
            
            # Synthesize
            audio = self._synthesize_text(sentence, streaming=(i == 0))
            
            if audio is not None:
                # Resample if needed
                if self.sample_rate != self.output_sample_rate:
                    audio = self._resample_audio(audio, self.sample_rate, self.output_sample_rate)
                
                # Queue audio for playback
                try:
                    self.audio_queue.put(audio, timeout=1)
                except queue.Full:
                    self.logger.warning("Audio queue full, dropping audio")
                
                self.total_audio_duration += len(audio) / self.output_sample_rate
        
        self.synthesis_count += 1
        self.logger.info(f"Synthesis complete for request {request_id}")
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences for streaming synthesis"""
        # Simple sentence splitting
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in '.!?' and len(current) > 10:
                sentences.append(current.strip())
                current = ""
        
        if current.strip():
            sentences.append(current.strip())
        
        return sentences
    
    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        if orig_sr == target_sr:
            return audio
        
        # Simple linear interpolation resampling
        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        
        indices = np.linspace(0, len(audio) - 1, target_length)
        resampled = np.interp(indices, np.arange(len(audio)), audio)
        
        return resampled
    
    def stop_playback(self):
        """Stop audio playback immediately"""
        self.is_playing = False
        
        # Clear audio queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break
        
        # Signal playback thread to stop
        self.audio_queue.put(None)
        
        # Stop any current playback
        try:
            sd.stop()
        except:
            pass
        
        # Wait for thread to finish
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1)
    
    def handle_control_message(self, msg: Dict[str, Any]):
        """Handle control messages"""
        msg_type = msg.get('type')
        
        if msg_type == 'abort':
            self.stop_playback()
            self.logger.info("TTS playback aborted")
            
        elif msg_type == 'config_reload':
            new_config = msg.get('config', {})
            new_language = new_config.get('tts_language')
            if new_language and new_language != self.language:
                self.language = new_language
    
    def start(self):
        """Main worker loop"""
        try:
            # Initialize TTS
            self.initialize_tts()
            
            # Report ready
            self.control_queue.put({
                'type': 'ready',
                'worker': 'tts',
                'timestamp': time.time()
            })
            
            self.logger.info("TTS worker started")
            
            last_metrics_time = time.time()
            
            # Main message handling loop
            while self.running:
                try:
                    # Check LLM queue
                    try:
                        llm_msg = self.queues['llm_to_tts'].get(timeout=0.1)
                        if llm_msg.get('type') == 'llm_response':
                            self.process_llm_response(llm_msg)
                    except:
                        pass
                    
                    # Check control queue
                    try:
                        control_msg = self.control_queue.get_nowait()
                        self.handle_control_message(control_msg)
                    except:
                        pass
                    
                    # Check performance
                    if self.first_chunk_latencies and len(self.first_chunk_latencies) >= 5:
                        avg_latency = sum(self.first_chunk_latencies[-5:]) / 5
                        if avg_latency > 30:  # Budget exceeded
                            self.logger.warning(f"First chunk latency {avg_latency:.1f}ms exceeds budget")
                    
                    # Log metrics every 30 seconds
                    if time.time() - last_metrics_time >= 30:
                        if self.first_chunk_latencies:
                            avg_first_chunk = sum(self.first_chunk_latencies) / len(self.first_chunk_latencies)
                        else:
                            avg_first_chunk = 0
                        
                        self.logger.info("TTS metrics", extra={
                            "stage": "tts",
                            "synthesis_count": self.synthesis_count,
                            "total_audio_minutes": self.total_audio_duration / 60,
                            "avg_first_chunk_ms": avg_first_chunk,
                            "engine": self.tts_engine_name
                        })
                        last_metrics_time = time.time()
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.logger.error(f"TTS worker error: {e}", exc_info=True)
                    time.sleep(0.1)
            
        except Exception as e:
            self.logger.error(f"TTS worker fatal error: {e}")
        finally:
            self.running = False
            self.stop_playback()
            self.logger.info("TTS worker stopped")