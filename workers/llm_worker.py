"""
LLM Worker - Large Language Model inference using vLLM
Target: ≤110ms prefill, ≤9ms per token
"""

import time
from typing import Optional, Dict, Any, List, Tuple
import multiprocessing as mp
import logging
import json
from collections import deque
import torch
import threading
import queue

# Conditional imports
try:
    from vllm import LLM, SamplingParams
    from vllm.utils import is_hip
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

class LLMWorker:
    """High-performance LLM inference with vLLM"""
    
    @staticmethod
    def run(config: Dict[str, Any], queues: Dict[str, mp.Queue]):
        """Main worker process entry point"""
        worker = LLMWorker(config, queues)
        worker.start()
    
    def __init__(self, config: Dict[str, Any], queues: Dict[str, mp.Queue]):
        self.config = config
        self.queues = queues
        self.logger = self._setup_logging()
        
        # Model configuration
        self.model_name = config.get('llm_model', 'meta-llama/Llama-3.2-3B-Instruct')
        self.max_tokens = config.get('llm_max_tokens', 512)
        self.temperature = config.get('llm_temperature', 0.7)
        self.gpu_memory_fraction = config.get('llm_gpu_memory_fraction', 0.4)
        
        # Check if vLLM is available
        self.use_vllm = VLLM_AVAILABLE and torch.cuda.is_available()
        
        # Model and sampling
        self.llm = None
        self.sampling_params = None
        
        # Conversation memory (last 4 turns)
        self.conversation_history = deque(maxlen=8)  # 4 user + 4 assistant
        self.kv_cache_reuse = True
        
        # Performance tracking
        self.request_count = 0
        self.prefill_latencies = []
        self.token_latencies = []
        self.total_tokens_generated = 0
        
        # Request queue for async processing
        self.request_queue = queue.Queue()
        self.current_request_id = None
        
        # Use dedicated control queue
        self.control_queue = queues['llm_control']
        self.running = True
          # System prompt
        self.system_prompt = """You are Wallie, a helpful and concise voice assistant. 
Keep responses brief and natural for spoken conversation. 
Avoid long explanations unless specifically asked.
Be friendly but efficient."""
        
    def _setup_logging(self) -> logging.Logger:
        """Setup worker-specific logging"""
        logger = logging.getLogger("wallie.llm")
        logger.setLevel(logging.INFO)
        return logger
    
    def initialize_llm(self):
        """Initialize LLM engine with memory-efficient fallback strategy"""
        try:
            self.logger.info(f"Loading LLM: {self.model_name}")
            
            # Force CPU model for memory efficiency if model is large
            large_models = ["llama", "falcon", "mistral", "mixtral"]
            is_large_model = any(model in self.model_name.lower() for model in large_models)
            
            if not self.use_vllm or is_large_model:
                self.logger.info("Using CPU model for memory efficiency")
                self._initialize_cpu_model()
                return
            
            # Only proceed with vLLM for small models
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=0.95,
                repetition_penalty=1.1
            )
            
            # Check available GPU memory before loading
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                free_memory = total_memory - torch.cuda.memory_allocated()
                required_memory = 2 * 1024**3  # Require at least 2GB free
                
                if free_memory < required_memory:
                    self.logger.warning(f"Insufficient GPU memory ({free_memory/1024**3:.1f}GB free), using CPU")
                    self._initialize_cpu_model()
                    return
              # Initialize vLLM with conservative settings
            self.llm = LLM(
                model=self.model_name,
                trust_remote_code=True,
                dtype="auto",
                gpu_memory_utilization=min(self.gpu_memory_fraction, 0.3),  # Cap at 30%
                max_model_len=1024,  # Reduced context
                enforce_eager=True,  # Avoid CUDA graphs for memory
                enable_prefix_caching=False,  # Disable caching to save memory
                disable_log_stats=True,
                tensor_parallel_size=1
            )
            
            self.logger.info("vLLM loaded successfully")
            
        except Exception as e:
            self.logger.error(f"vLLM initialization failed: {e}")            # Clear any partial GPU allocations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._initialize_cpu_model()
    
    def _initialize_cpu_model(self):
        """Initialize CPU-based model as fallback"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.logger.info("Initializing CPU model with transformers")
            self.use_vllm = False
            
            # Use the original lightweight model
            cpu_model_name = self.model_name
            
            # Simple direct model loading approach
            self.tokenizer = AutoTokenizer.from_pretrained(cpu_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.cpu_model = AutoModelForCausalLM.from_pretrained(
                cpu_model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            self.cpu_model.to("cpu")
            
            self.logger.info("CPU model ready")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize CPU model: {e}")
            raise RuntimeError("No LLM backend available")
    
    def _build_prompt(self, user_text: str) -> str:
        """Build prompt with conversation history"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history
        for i, msg in enumerate(self.conversation_history):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": msg})
        
        # Add current user message
        messages.append({"role": "user", "content": user_text})
        
        # Format for Llama-3 chat template
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
            elif msg["role"] == "user":
                prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
            else:
                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
        
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return prompt
    
    def _generate(self, user_text: str, use_history: bool = True) -> Tuple[str, Dict[str, float]]:
        """Generate response with performance tracking"""
        start_time = time.perf_counter()
        
        # Build prompt
        if use_history:
            prompt = self._build_prompt(user_text)
        else:
            prompt = user_text
        
        # Track prefill start
        prefill_start = time.perf_counter()
        
        if self.use_vllm and self.llm:
            # Generate with vLLM
            outputs = self.llm.generate(
                [prompt],
                self.sampling_params,
                use_tqdm=False
            )
            
            # Get response
            response = outputs[0].outputs[0].text.strip()
            
            # Calculate metrics
            prefill_time = (time.perf_counter() - prefill_start) * 1000
            total_time = (time.perf_counter() - start_time) * 1000
            num_tokens = len(outputs[0].outputs[0].token_ids)
            
        else:
            # CPU generation
            response = self._generate_cpu(prompt)
            prefill_time = (time.perf_counter() - prefill_start) * 1000
            total_time = (time.perf_counter() - start_time) * 1000
            num_tokens = len(response.split())  # Approximate
        
        metrics = {
            "prefill_ms": prefill_time,
            "total_ms": total_time,
            "tokens": num_tokens,
            "tokens_per_sec": num_tokens / (total_time / 1000) if total_time > 0 else 0        }
        
        # Track performance
        self.prefill_latencies.append(prefill_time)
        self.total_tokens_generated += num_tokens
        
        return response, metrics
    
    def _generate_cpu(self, prompt: str) -> str:
        """Generate using CPU model"""
        try:
            # Use direct model approach (simpler and more reliable)
            if hasattr(self, 'cpu_model') and hasattr(self, 'tokenizer'):
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                
                with torch.no_grad():
                    outputs = self.cpu_model.generate(
                        inputs.input_ids,
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature,
                        do_sample=True,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.eos_token_id,
                        attention_mask=inputs.get('attention_mask', None)
                    )
                
                response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                return response.strip()
                
        except Exception as e:
            self.logger.error(f"CPU generation failed: {e}")
        
        # Ultimate fallback response
        return "I'm sorry, I'm having trouble processing your request right now."
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single LLM request"""
        try:
            user_text = request.get('text', '')
            request_id = request.get('id', time.time())
            
            self.logger.info(f"Processing: {user_text[:50]}...")
            
            # Generate response
            response, metrics = self._generate(user_text)
            
            # Update conversation history
            self.conversation_history.append(user_text)
            self.conversation_history.append(response)
            
            self.request_count += 1
            
            # Log performance
            self.logger.info(f"Generated {metrics['tokens']} tokens in {metrics['total_ms']:.1f}ms", extra={
                "stage": "llm",
                "latency_ms": metrics['total_ms'],
                "prefill_ms": metrics['prefill_ms'],
                "tokens_per_sec": metrics['tokens_per_sec']
            })
            
            return {
                'type': 'llm_response',
                'text': response,
                'timestamp': time.time(),
                'request_id': request_id,
                'metrics': metrics
            }
            
        except torch.cuda.OutOfMemoryError:
            self.logger.error("GPU OOM during generation")
            torch.cuda.empty_cache()
            
            # Retry with shorter context
            if len(self.conversation_history) > 2:
                self.conversation_history = deque(list(self.conversation_history)[-2:], maxlen=8)
                return self.process_request(request)
            
            return {
                'type': 'llm_error',
                'error': 'GPU memory error',
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"LLM generation error: {e}")
            return {
                'type': 'llm_error',
                'error': str(e),
                'timestamp': time.time()
            }
    
    def handle_asr_message(self, msg: Dict[str, Any]):
        """Handle transcription from ASR"""
        if msg.get('type') != 'transcription':
            return
        
        text = msg.get('text', '').strip()
        is_final = msg.get('is_final', False)
        
        if not text:
            return
        
        if is_final:
            # Process complete utterance
            request = {
                'text': text,
                'id': time.time(),
                'timestamp': msg.get('timestamp')
            }
            
            # Process immediately
            response = self.process_request(request)
            
            # Send to TTS
            if response.get('type') == 'llm_response':
                try:
                    self.queues['llm_to_tts'].put_nowait(response)
                except:
                    self.logger.warning("TTS queue full")
    
    def handle_control_message(self, msg: Dict[str, Any]):
        """Handle control messages"""
        msg_type = msg.get('type')
        
        if msg_type == 'abort':
            # Cancel current generation if possible
            self.current_request_id = None
            self.logger.info("LLM generation aborted")
            
        elif msg_type == 'config_reload':
            new_config = msg.get('config', {})
            self.temperature = new_config.get('llm_temperature', self.temperature)
            self.max_tokens = new_config.get('llm_max_tokens', self.max_tokens)
            
            # Update sampling params if using vLLM
            if self.use_vllm and self.sampling_params:
                self.sampling_params = SamplingParams(
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=0.95,
                    repetition_penalty=1.1
                )
    
    def check_performance(self):
        """Check if performance targets are met"""
        if len(self.prefill_latencies) < 5:
            return
        
        recent_prefill = self.prefill_latencies[-5:]
        avg_prefill = sum(recent_prefill) / len(recent_prefill)
        
        if avg_prefill > 110:  # Prefill budget exceeded
            self.logger.warning(f"Prefill latency {avg_prefill:.1f}ms exceeds budget")
            
            # Reduce context length
            if len(self.conversation_history) > 4:
                self.conversation_history = deque(list(self.conversation_history)[-4:], maxlen=8)
                self.logger.info("Reduced conversation history for performance")
    
    def start(self):
        """Main worker loop"""
        try:
            # Initialize LLM
            self.initialize_llm()
            
            # Report ready
            self.control_queue.put({
                'type': 'ready',
                'worker': 'llm',
                'timestamp': time.time()
            })
            
            self.logger.info("LLM worker started")
            
            last_metrics_time = time.time()
            
            # Main message handling loop
            while self.running:
                try:
                    # Check ASR queue with timeout
                    try:
                        asr_msg = self.queues['asr_to_llm'].get(timeout=0.1)
                        self.handle_asr_message(asr_msg)
                    except:
                        pass
                    
                    # Check control queue
                    try:
                        control_msg = self.control_queue.get_nowait()
                        self.handle_control_message(control_msg)
                    except:
                        pass
                    
                    # Check performance periodically
                    self.check_performance()
                    
                    # Log metrics every 30 seconds
                    if time.time() - last_metrics_time >= 30:
                        if self.prefill_latencies:
                            avg_prefill = sum(self.prefill_latencies) / len(self.prefill_latencies)
                        else:
                            avg_prefill = 0
                            
                        tokens_per_request = self.total_tokens_generated / self.request_count if self.request_count > 0 else 0
                        
                        self.logger.info("LLM metrics", extra={
                            "stage": "llm",
                            "request_count": self.request_count,
                            "avg_prefill_ms": avg_prefill,
                            "avg_tokens_per_request": tokens_per_request,
                            "total_tokens": self.total_tokens_generated
                        })
                        last_metrics_time = time.time()
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.logger.error(f"LLM worker error: {e}", exc_info=True)
                    time.sleep(0.1)
            
        except Exception as e:
            self.logger.error(f"LLM worker fatal error: {e}")
        finally:
            self.running = False
            self.logger.info("LLM worker stopped")