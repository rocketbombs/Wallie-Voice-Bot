"""
ASR Worker - Streaming speech recognition using faster-whisper
Target: ≤90ms to first partial result
"""

import time
import numpy as np
from typing import Optional, Dict, Any, List
import multiprocessing as mp
import logging
from faster_whisper import WhisperModel
import queue as Queue  # Add at top of file with other imports
import threading
from collections import deque
import asyncio

class ASRWorker:
    """Streaming ASR with faster-whisper"""
    
    @staticmethod
    def run(config: Dict[str, Any], queues: Dict[str, mp.Queue]):
        """Main worker process entry point"""
        worker = ASRWorker(config, queues)
        worker.start()
    
    def __init__(self, config: Dict[str, Any], queues: Dict[str, mp.Queue]):
        self.config = config
        self.queues = queues
        self.logger = self._setup_logging()
        
        # Model configuration
        self.model_size = config.get('asr_model', 'tiny.en')
        self.device = config.get('asr_device', 'cuda')
        self.compute_type = config.get('asr_compute_type', 'float16')
        
        # Streaming configuration
        self.sample_rate = 16000
        self.chunk_length = 1.0  # Process 1s chunks
        self.chunk_samples = int(self.sample_rate * self.chunk_length)
        self.overlap = 0.1  # 100ms overlap between chunks
        self.overlap_samples = int(self.sample_rate * self.overlap)
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=self.sample_rate * 30)  # 30s max
        self.processing_queue = Queue.Queue()
        
        # Model instance
        self.model: Optional[WhisperModel] = None
        
        # Performance tracking
        self.transcription_count = 0
        self.total_audio_processed = 0.0
        self.first_partial_latencies = []
        
        # Precision downgrade flag
        self.precision_downgraded = False
        
        self.control_queue = queues['asr_control']  # Use dedicated control queue
        self.running = True  # Add process state
        
    def _setup_logging(self) -> logging.Logger:
        """Setup worker-specific logging"""
        logger = logging.getLogger("wallie.asr")
        logger.setLevel(logging.INFO)
        return logger
    
    def initialize_model(self):
        """Initialize faster-whisper model with CUDA fallback"""
        try:
            self.logger.info(f"Loading Whisper model: {self.model_size}")
            
            # Try with requested configuration
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=4,
                num_workers=2
            )
            
            # Warm up model
            self.logger.info("Warming up ASR model...")
            dummy_audio = np.zeros(self.sample_rate * 2, dtype=np.float32)
            segments, _ = self.model.transcribe(
                dummy_audio,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                no_speech_threshold=0.6
            )
            list(segments)  # Force evaluation
            
            self.logger.info(f"ASR model ready ({self.device}, {self.compute_type})")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model with {self.device}: {e}")
            
            # Try CPU fallback
            if self.device != 'cpu':
                self.logger.warning("Falling back to CPU")
                self.device = 'cpu'
                self.compute_type = 'int8'
                self.model = WhisperModel(
                    self.model_size,
                    device='cpu',
                    compute_type='int8',
                    cpu_threads=8
                )
    
    def process_audio_chunk(self, audio_data: np.ndarray, is_first: bool = False) -> Optional[str]:
        """Process a single audio chunk and return transcription"""
        start_time = time.perf_counter()
        
        try:
            # Ensure float32
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)
            
            # Run transcription
            segments, info = self.model.transcribe(
                audio_float,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=100
                )
            )
            
            # Extract text
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())
            
            text = ' '.join(text_parts)
            
            # Track first partial latency
            if is_first and text:
                latency_ms = (time.perf_counter() - start_time) * 1000
                self.first_partial_latencies.append(latency_ms)
                self.logger.info(f"First partial in {latency_ms:.1f}ms: {text[:50]}...")
            
            return text
            
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            return None
    
    def audio_processing_thread(self):
        """Background thread for processing audio chunks"""
        overlapping_audio = np.array([], dtype=np.float32)
        
        while True:
            try:
                # Get audio chunk from queue
                chunk_data = self.processing_queue.get(timeout=1.0)
                
                if chunk_data is None:  # Shutdown signal
                    break
                
                audio_chunk = chunk_data['audio']
                is_first = chunk_data.get('is_first', False)
                
                # Combine with overlap from previous chunk
                if len(overlapping_audio) > 0:
                    audio_to_process = np.concatenate([overlapping_audio, audio_chunk])
                else:
                    audio_to_process = audio_chunk
                
                # Keep overlap for next chunk
                if len(audio_to_process) > self.overlap_samples:
                    overlapping_audio = audio_to_process[-self.overlap_samples:]
                
                # Process chunk
                text = self.process_audio_chunk(audio_to_process, is_first)
                
                if text:
                    # Send to LLM
                    try:
                        self.queues['asr_to_llm'].put_nowait({
                            'type': 'transcription',
                            'text': text,
                            'timestamp': time.time(),
                            'is_partial': True
                        })
                    except:
                        pass
                
            except Queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Audio processing error: {e}")
    
    def handle_vad_message(self, msg: Dict[str, Any]):
        """Handle messages from VAD worker"""
        msg_type = msg.get('type')
        
        if msg_type == 'audio_chunk':
            # Add to buffer
            audio_data = msg.get('data')
            is_first = msg.get('is_first', False)
            
            self.audio_buffer.extend(audio_data)
            
            # Check if we have enough for a chunk
            if len(self.audio_buffer) >= self.chunk_samples:
                # Extract chunk
                chunk = np.array(list(self.audio_buffer)[:self.chunk_samples])
                
                # Remove processed samples (keeping overlap)
                for _ in range(self.chunk_samples - self.overlap_samples):
                    try:
                        self.audio_buffer.popleft()
                    except:
                        break
                
                # Queue for processing
                self.processing_queue.put({
                    'audio': chunk,
                    'is_first': is_first
                })
        
        elif msg_type == 'end_of_speech':
            # Process any remaining audio
            if len(self.audio_buffer) > 0:
                remaining = np.array(list(self.audio_buffer))
                self.audio_buffer.clear()
                
                # Process final chunk
                text = self.process_audio_chunk(remaining)
                
                if text:
                    # Send final transcription
                    try:
                        self.queues['asr_to_llm'].put({
                            'type': 'transcription',
                            'text': text,
                            'timestamp': time.time(),
                            'is_partial': False,
                            'is_final': True
                        })
                    except:
                        pass
                
                self.transcription_count += 1
                self.total_audio_processed += msg.get('duration', 0)
    
    async def handle_control_messages(self):
        """Async handler for control messages"""
        while True:
            try:
                msg = await self.get_control_message()
                if msg:
                    msg_type = msg.get('type')
                    
                    if msg_type == 'abort':
                        await self.handle_abort()
                    elif msg_type == 'config_reload':
                        await self.handle_config_reload(msg.get('config', {}))
            except Exception as e:
                self.logger.error(f"Control message error: {e}")
            await asyncio.sleep(0.01)
    
    async def get_control_message(self) -> Optional[Dict[str, Any]]:
        """Non-blocking control message check"""
        try:
            return self.control_queue.get_nowait()
        except Queue.Empty:
            return None
        
    async def handle_abort(self):
        """Handle abort request"""
        self.audio_buffer.clear()
        while not self.processing_queue.empty():
            try:
                self.processing_queue.get_nowait()
            except:
                break
        self.logger.info("ASR aborted")
    
    async def handle_config_reload(self, new_config: Dict[str, Any]):
        """Handle config reload"""
        new_model = new_config.get('asr_model')
        if new_model and new_model != self.model_size:
            self.model_size = new_model
            await self.initialize_model()
    
    def downgrade_precision(self):
        """Downgrade model precision for performance"""
        if not self.precision_downgraded:
            self.logger.warning("Downgrading ASR precision for performance")
            self.precision_downgraded = True
            
            if self.compute_type == 'float16':
                self.compute_type = 'int8'
            elif self.device == 'cuda':
                self.device = 'cpu'
                self.compute_type = 'int8'
            
            # Reinitialize model
            self.initialize_model()
    
    def start(self):
        """Main worker loop with async support"""
        try:
            # Initialize model
            self.initialize_model()
            
            # Start processing thread
            processor_thread = threading.Thread(
                target=self.audio_processing_thread,
                name="asr_processor"
            )
            processor_thread.daemon = True  # Ensure thread exits with process
            processor_thread.start()
            
            # Initialize asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Start async tasks
            tasks = [
                loop.create_task(self.handle_control_messages()),
                loop.create_task(self.process_audio_stream())
            ]
            
            # Signal ready
            self.control_queue.put({
                'type': 'ready',
                'worker': 'asr',
                'timestamp': time.time()
            })
            
            # Run until stopped
            while self.running:
                try:
                    loop.run_until_complete(asyncio.gather(*tasks))
                except asyncio.CancelledError:
                    break
            
        except Exception as e:
            self.logger.error(f"ASR worker error: {e}")
        finally:
            self.running = False
            self.cleanup()
    
    async def process_audio_stream(self):
        """Process audio stream with metrics"""
        last_metrics_time = time.time()
        
        while True:
            try:
                # Process messages from VAD
                try:
                    msg = self.queues['vad_to_asr'].get_nowait()
                    if msg:
                        self.handle_vad_message(msg)
                except queue.Empty:
                    pass
                
                # Log metrics every 30 seconds
                if time.time() - last_metrics_time >= 30:
                    self.logger.info("ASR metrics", extra={
                        'stage': 'asr',
                        'transcription_count': self.metrics['transcription_count'],
                        'audio_processed_sec': self.metrics['audio_processed_sec'],
                        'avg_first_partial_ms': (
                            sum(self.metrics['first_partial_latency_ms'][-5:]) / 
                            len(self.metrics['first_partial_latency_ms'][-5:])
                            if self.metrics['first_partial_latency_ms'] else 0
                        )
                    })
                    last_metrics_time = time.time()
                    
                await asyncio.sleep(0.01)  # Prevent CPU spin
                
            except Exception as e:
                self.logger.error(f"Audio stream processing error: {e}")
                await asyncio.sleep(0.1)  # Back off on error
    
    def cleanup(self):
        """Clean up resources"""
        try:
            # Signal processing thread to stop
            self.running = False
            if hasattr(self, 'processing_queue'):
                self.processing_queue.put(None)
            
            # Clear buffers
            self.audio_buffer.clear()
            
            # Clean up model
            if self.model is not None:
                del self.model
            
            # Log final metrics
            if hasattr(self, 'metrics'):
                self.logger.info("ASR final metrics", extra={
                    'stage': 'asr',
                    'total_transcriptions': self.metrics['transcription_count'],
                    'total_audio_processed': self.metrics['audio_processed_sec'],
                    'avg_first_partial_ms': (
                        sum(self.metrics['first_partial_latency_ms']) / 
                        len(self.metrics['first_partial_latency_ms'])
                        if self.metrics['first_partial_latency_ms'] else 0
                    )
                })
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")