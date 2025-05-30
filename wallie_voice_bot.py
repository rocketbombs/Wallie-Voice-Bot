#!/usr/bin/env python3
"""
Wallie Voice Bot - Production-grade offline voice assistant daemon
Target: ≤250ms end-to-end latency from speech end to first audio
"""

import os
# Fix OpenMP conflict issues with multiple ML libraries
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import asyncio
import signal
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
import multiprocessing as mp
import numpy as np
import typer
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

# Type alias to help with static analysis
ConfigBaseClass = BaseSettings
import toml
import logging
import json
from datetime import datetime
import psutil
import queue as Queue  # Add at top with other imports

# Try importing torch for GPU monitoring
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import worker modules
from workers.asr_worker import ASRWorker
from workers.llm_worker import LLMWorker
from workers.tts_worker import TTSWorker
from workers.vad_worker import VADWorker

# Configure logging
class JSONLFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "stage": getattr(record, "stage", "main"),
            "message": record.getMessage(),
            "latency_ms": getattr(record, "latency_ms", None),
            "gpu_mem_mb": getattr(record, "gpu_mem_mb", None),
            "audio_overruns": getattr(record, "audio_overruns", 0)
        }
        return json.dumps(log_obj)

# Performance decorator
class StageOverrunError(Exception):
    pass

def stage_timer(stage_name: str, budget_ms: float):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000
                
                logger = logging.getLogger(__name__)
                logger.info(f"{stage_name} completed", extra={
                    "stage": stage_name,
                    "latency_ms": elapsed_ms
                })
                
                if elapsed_ms > budget_ms:
                    self = args[0] if args else None
                    if self is not None and hasattr(self, '_overrun_count'):
                        self._overrun_count[stage_name] = self._overrun_count.get(stage_name, 0) + 1
                        if self._overrun_count[stage_name] >= 2:
                            raise StageOverrunError(f"{stage_name} exceeded {budget_ms}ms budget twice")
                
                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger = logging.getLogger(__name__)
                logger.error(f"{stage_name} failed after {elapsed_ms}ms: {e}")
                raise
        return wrapper
    return decorator

class WallieConfig(BaseSettings):  # type: ignore
    """Configuration loaded from ~/.wallie_voice_bot/config.toml"""
    wake_word: str = "wallie"
    wake_word_sensitivity: float = 0.7
    
    asr_model: str = "tiny.en"
    asr_device: str = "cpu"  # Use CPU to conserve GPU memory for LLM
    asr_compute_type: str = "float32"
    
    llm_model: str = "microsoft/DialoGPT-small"  # Much smaller model (~117MB vs 3GB)
    llm_max_tokens: int = 100  # Reduced for faster responses
    llm_temperature: float = 0.7
    llm_gpu_memory_fraction: float = 0.2  # Reduced GPU memory usage
    
    tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts_speaker_wav: Optional[str] = None
    tts_language: str = "en"
    
    audio_sample_rate: int = 16000
    audio_chunk_size: int = 512
    
    watchdog_interval_sec: int = 2
    watchdog_max_restarts: int = 3
    
    enable_prometheus: bool = False
    prometheus_port: int = 9090
    
    archive_transcripts: bool = False
    log_retention_hours: int = 24
    
    class Config:
        env_prefix = "WALLIE_"
        extra = "allow"

class WallieDaemon:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Process management
        self.workers: Dict[str, mp.Process] = {}
        self.queues: Dict[str, mp.Queue] = {}
        self.running = False
        self._overrun_count: Dict[str, int] = {}
        
        # Pipeline queues
        self.queues['vad_to_asr'] = mp.Queue(maxsize=10)
        self.queues['asr_to_llm'] = mp.Queue(maxsize=10)
        self.queues['llm_to_tts'] = mp.Queue(maxsize=10)
        
        # Dedicated control queues
        self.queues['vad_control'] = mp.Queue()
        self.queues['asr_control'] = mp.Queue()
        self.queues['llm_control'] = mp.Queue()
        self.queues['tts_control'] = mp.Queue()
        
        # Performance metrics
        self.metrics = {
            'cold_start_time': None,
            'session_count': 0,
            'total_latency_ms': []
        }
        
    def _load_config(self) -> WallieConfig:
        """Load configuration with hot-reload support"""
        if self.config_path.exists():
            config_dict = toml.load(self.config_path)
            return WallieConfig(**config_dict)
        return WallieConfig()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup JSONL logging with rotation"""
        log_dir = Path.home() / ".wallie_voice_bot" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger = logging.getLogger("wallie")
        logger.setLevel(logging.INFO)
        
        # JSONL file handler with daily rotation
        from logging.handlers import TimedRotatingFileHandler
        handler = TimedRotatingFileHandler(
            log_dir / "wallie.jsonl",
            when="midnight",
            interval=1,
            backupCount=self.config.log_retention_hours // 24
        )
        handler.setFormatter(JSONLFormatter())
        logger.addHandler(handler)
        
        return logger
    
    def _get_worker_config(self, worker_name: str) -> Dict[str, Any]:
        """Get worker-specific configuration"""
        config = self.config.dict()
        
        # Add worker-specific settings
        if worker_name == 'vad':
            config.update({
                'audio_sample_rate': 16000,
                'frame_length': 512
            })
        elif worker_name == 'asr':
            config.update({
                'max_audio_length': 30.0,
                'chunk_overlap': 0.1
            })
        elif worker_name == 'llm':
            config.update({
                'max_history_turns': 4,
                'temperature': self.config.llm_temperature
            })
        elif worker_name == 'tts':
            config.update({
                'output_sample_rate': 48000,
                'enable_caching': True
            })
        
        return config

    def _start_worker(self, name: str, worker_class: type, *args):
        """Start a worker process with worker-specific config"""
        if name in self.workers and self.workers[name].is_alive():
            self.workers[name].terminate()
            self.workers[name].join(timeout=2)
        
        worker_config = self._get_worker_config(name)
        
        process = mp.Process(
            target=worker_class.run,
            args=(worker_config, self.queues, *args),
            name=f"wallie_{name}"
        )
        process.start()
        self.workers[name] = process
        self.logger.info(f"Started {name} worker", extra={"stage": name})
    
    @stage_timer("initialization", 10000)  # 10s cold start budget
    async def initialize(self):
        """Initialize all workers and verify GPU availability"""
        start_time = time.perf_counter()
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.logger.info(f"GPU detected: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f}GB)")
            else:
                self.logger.warning("No GPU detected, falling back to CPU")
                # Update config for CPU mode
                self.config.asr_device = 'cpu'
                self.config.asr_compute_type = 'int8'
        except ImportError:
            self.logger.warning("PyTorch not installed, running in CPU mode")
        
        # Start workers with dependency order and health checks
        workers_to_start = [
            ("vad", VADWorker),
            ("asr", ASRWorker),
            ("llm", LLMWorker),
            ("tts", TTSWorker)
        ]
        
        ready_workers = set()
        for name, worker_class in workers_to_start:
            try:
                self._start_worker(name, worker_class)
                await asyncio.sleep(0.5)  # Allow worker to initialize
                
                # Wait for ready signal from worker-specific control queue
                start = time.perf_counter()
                while time.perf_counter() - start < 8.0:
                    try:
                        msg = self.queues[f'{name}_control'].get_nowait()  # Updated queue name
                        if msg.get('type') == 'ready' and msg.get('worker') == name:
                            ready_workers.add(name)
                            self.logger.info(f"{name} ready")
                            break
                    except:
                        await asyncio.sleep(0.1)
            
            except Exception as e:
                self.logger.error(f"Failed to start {name}: {e}")
                raise RuntimeError(f"Critical worker {name} failed to start")
    
        if len(ready_workers) < len(workers_to_start):
            missing = set(w[0] for w in workers_to_start) - ready_workers
            raise RuntimeError(f"Workers failed to initialize: {missing}")
        
        self.metrics['cold_start_time'] = time.perf_counter() - start_time
        self.logger.info(f"Cold start completed in {self.metrics['cold_start_time']:.1f}s")
    
    async def watchdog_loop(self):
        """Monitor worker health and restart if needed"""
        restart_counts = {}
        
        while self.running:
            await asyncio.sleep(self.config.watchdog_interval_sec)
            
            for name, process in self.workers.items():
                if not process.is_alive():
                    restart_counts[name] = restart_counts.get(name, 0) + 1
                    
                    if restart_counts[name] > self.config.watchdog_max_restarts:
                        self.logger.error(f"{name} worker exceeded max reboots")
                        continue
                    
                    self.logger.warning(f"Restarting {name} worker (attempt {restart_counts[name]})")
                    
                    # Exponential backoff
                    await asyncio.sleep(2 ** restart_counts[name])
                    
                    # Restart worker
                    worker_class = {
                        "vad": VADWorker,
                        "asr": ASRWorker,
                        "llm": LLMWorker,
                        "tts": TTSWorker
                    }.get(name)
                    
                    if worker_class:
                        self._start_worker(name, worker_class)
    
    async def handle_wake_word_interrupt(self):
        """Handle wake word interrupts with dedicated control queues"""
        while self.running:
            try:
                msg = await self.get_wake_word_message()
                if msg and msg.get('type') == 'wake_word_detected':
                    self.logger.info("Wake word interrupt received")
                    
                    # Send abort to all workers through their control queues
                    abort_msg = {'type': 'abort', 'timestamp': time.time()}
                    control_queues = [
                        'vad_control', 'asr_control', 
                        'llm_control', 'tts_control'
                    ]
                    
                    for queue_name in control_queues:
                        try:
                            self.queues[queue_name].put_nowait(abort_msg)
                        except Queue.Full:
                            self.logger.error(f"Failed to send abort to {queue_name}")
                      # Clear pipeline queues
                    await self.clear_pipeline_queues()
                    
                    self.metrics['session_count'] += 1
                    
            except Exception as e:
                if not isinstance(e, asyncio.TimeoutError):
                    self.logger.error(f"Wake word interrupt error: {e}")
                await asyncio.sleep(0.05)

    async def reload_config(self):
        """Hot-reload configuration on SIGHUP"""
        self.logger.info("Reloading configuration")
        new_config = self._load_config()
        if new_config.dict() != self.config.dict():
            self.config = new_config
            for queue in self.queues.values():
                if isinstance(queue, mp.Queue):
                    try:
                        queue.put_nowait({'type': 'config_reload', 'config': self.config.dict()})
                    except Queue.Full:
                        pass
    
    async def clear_pipeline_queues(self):
        """Clear all pipeline queues"""
        pipeline_queues = [
            'vad_to_asr', 'asr_to_llm', 
            'llm_to_tts', 'tts_to_audio'
        ]
        for queue_name in pipeline_queues:
            if queue_name in self.queues:
                while True:
                    try:
                        self.queues[queue_name].get_nowait()
                    except (Queue.Empty, KeyError):                        break

    async def run(self):
        """Main daemon loop"""
        self.running = True

        def signal_handler(sig, frame):
            self.logger.info(f"Received signal {sig}")
            try:
                loop = asyncio.get_running_loop()
                # Check for SIGHUP safely for reload on Unix systems
                sighup_value = getattr(signal, 'SIGHUP', None)
                if sighup_value is not None and sig == sighup_value:
                    loop.create_task(self.reload_config())
                else:
                    self.running = False
            except RuntimeError:
                # No running event loop
                self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if sys.platform != "win32" and hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, signal_handler)
        
        try:
            # Initialize system
            await self.initialize()
            
            # Start concurrent tasks
            tasks = [
                asyncio.create_task(self.watchdog_loop()),
                asyncio.create_task(self.handle_wake_word_interrupt())
            ]
            
            # Wait for shutdown
            while self.running:
                await asyncio.sleep(1)
                
                # Log system metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                self.logger.info("System metrics", extra={
                    "stage": "main",
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "gpu_mem_mb": self._get_gpu_memory_usage()
                })
            
        finally:
            # Cleanup
            self.logger.info("Shutting down")
            
            # Cancel tasks
            for task in tasks:
                task.cancel()
              # Terminate workers
            for name, process in self.workers.items():
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2)
                    if process.is_alive():
                        process.kill()
            
            self.logger.info("Shutdown complete")
    
    def _get_gpu_memory_usage(self) -> Optional[int]:
        """Get current GPU memory usage in MB"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return int(torch.cuda.memory_allocated() / 1024**2)
        except:
            pass
        return None

    async def get_wake_word_message(self) -> Optional[Dict[str, Any]]:
        """Get wake word message with timeout"""
        try:
            loop = asyncio.get_running_loop()
            msg = await loop.run_in_executor(
                None,
                lambda: self.queues['vad_control'].get(timeout=0.1)
            )
            return msg
        except Queue.Empty:
            return None

app = typer.Typer()

@app.command()
def main(
    config: Path = typer.Option(
        Path.home() / ".wallie_voice_bot" / "config.toml",
        "--config", "-c",
        help="Configuration file path"
    ),
    daemon: bool = typer.Option(
        False,
        "--daemon", "-d",
        help="Run as daemon"
    )
):
    """Wallie Voice Bot - Offline GPT-grade voice assistant"""
    
    # Create config directory if needed
    config.parent.mkdir(parents=True, exist_ok=True)
    
    # Create default config if missing
    if not config.exists():
        default_config = WallieConfig()
        with open(config, "w") as f:
            toml.dump(default_config.dict(), f)
        print(f"Created default config at {config}")
    
    # Daemonize if requested
    if daemon and sys.platform != "win32":
        try:
            import daemon
            from daemon import pidfile
            pid_file = Path.home() / ".wallie_voice_bot" / "wallie.pid"
            with daemon.DaemonContext(
                working_directory='/',
                umask=0o002,
                pidfile=pidfile.TimeoutPIDLockFile(str(pid_file)),
                detach_process=True
            ):
                wallie = WallieDaemon(config)
                asyncio.run(wallie.run())
        except ImportError:
            print("python-daemon not installed, running in foreground")
            wallie = WallieDaemon(config)
            asyncio.run(wallie.run())
    else:
        wallie = WallieDaemon(config)
        asyncio.run(wallie.run())

def setup_asyncio():
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop

# In your main() or __main__ section:
if __name__ == "__main__":
    loop = setup_asyncio()
    try:
        app()
    finally:
        loop.close()
