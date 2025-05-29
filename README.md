# Wallie Voice Bot

A production-grade, privacy-first voice assistant that runs entirely offline on your local machine. Wallie achieves sub-250ms response times from speech end to first spoken word, rivaling cloud-based assistants while keeping all your data private.

---

## Features

- **Completely Offline**: All inference happens locally—no internet required.
- **Ultra-Low Latency**: ≤250ms from end of speech to first audible response.
- **Production-Ready**: Self-healing daemon with process isolation and automatic recovery.
- **Privacy-First**: Audio and transcripts never leave your machine.
- **High Quality**: GPT-grade responses using Llama 3.2 models.
- **Natural Voice**: Streaming TTS with XTTS v2 for human-like speech.
- **Customizable**: Hot-reloadable configuration and voice cloning support.

---

## Performance Targets

| Stage             | Target Latency | Description                        |
|-------------------|---------------|------------------------------------|
| ASR First Partial | ≤90ms         | Time to first transcription chunk  |
| LLM Prefill       | ≤110ms        | Time to process prompt             |
| LLM Per Token     | ≤9ms          | Generation speed per token         |
| TTS First Chunk   | ≤30ms         | Time to first audio output         |
| **Total E2E**     | **≤250ms**    | **Speech end to first audio**      |

---

## System Requirements

**Hardware**
- **GPU**: NVIDIA RTX 3080 (10GB) or better
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 20GB for models
- **Audio**: USB microphone and speakers

**Software**
- **OS**: Ubuntu 22.04 LTS or Windows 11 (via WSL2)
- **Python**: 3.12+
- **CUDA**: 12.4+ with latest drivers
- **NVIDIA Driver**: 545.0+

---

## Quick Start

```bash
git clone https://github.com/yourusername/wallie-voice-bot.git
cd wallie-voice-bot
chmod +x scripts/install_deps.sh
./scripts/install_deps.sh
python wallie_voice_bot.py
```

---

## Configuration

Wallie creates a default config at `~/.wallie_voice_bot/config.toml` if missing. All options can also be set via environment variables (prefix: `WALLIE_`).

Example:

```toml
wake_word = "wallie"
wake_word_sensitivity = 0.7
asr_model = "tiny.en"
llm_model = "meta-llama/Llama-3.2-3B-Instruct"
tts_speaker_wav = "/path/to/voice.wav"
llm_gpu_memory_fraction = 0.4
```

---

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Microphone │────▶│  VAD/Wake   │────▶│     ASR     │────▶│     LLM     │
│   (48kHz)   │     │  (Porcupine)│     │  (Whisper)  │     │   (vLLM)    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                            │                                         │
                            │ Interrupt                              ▼
                            │                               ┌─────────────┐
                            └──────────────────────────────│     TTS     │
                                                           │  (XTTS v2)  │
                                                           └─────────────┘
                                                                    │
                                                                    ▼
                                                           ┌─────────────┐
                                                           │  Speakers   │
                                                           │   (48kHz)   │
                                                           └─────────────┘
```

Each component runs in its own process with Queue-based IPC:
- **VAD Worker**: Wake word detection and audio capture
- **ASR Worker**: Streaming speech recognition
- **LLM Worker**: Text generation with conversation memory
- **TTS Worker**: Speech synthesis and playback

---

## Self-Healing Features

- Automatic process restart with exponential backoff
- GPU OOM recovery with precision downgrade
- Dynamic performance adaptation
- Hot configuration reload (SIGHUP)

---

## Advanced Usage

### Custom Wake Words

1. Create a custom wake word at [Picovoice Console](https://console.picovoice.ai)
2. Download the `.ppn` file
3. Place it in `~/.wallie_voice_bot/wake_words/`
4. Update config: `wake_word = "your_word"`

### Voice Cloning

1. Record 10–30 seconds of clear speech
2. Save as WAV file (mono, 22050Hz+)
3. Set path in config: `tts_speaker_wav = "/path/to/voice.wav"`

### Performance Tuning

For GPU memory issues:
```toml
llm_gpu_memory_fraction = 0.3
asr_compute_type = "int8"
```
For faster responses:
```toml
asr_model = "tiny.en"
llm_max_tokens = 256
```

---

## Monitoring

Enable Prometheus metrics:
```toml
enable_prometheus = true
prometheus_port = 9090
```

View logs:
```bash
tail -f ~/.wallie_voice_bot/logs/wallie.jsonl | jq
cat ~/.wallie_voice_bot/logs/wallie.jsonl | jq 'select(.stage=="asr")'
```

---

## Troubleshooting

**PyAudio Build Error**
```bash
sudo apt-get install python3-dev python3.12-dev portaudio19-dev
pip install PyAudio
```
Or skip PyAudio (sounddevice is sufficient):
```bash
pip install --no-deps pyaudio || echo "Skipping PyAudio"
```

**Daemon Mode**
- `python-daemon` is optional. If not installed, Wallie runs in the foreground.

**No Wake Word Detection**
- Check microphone permissions and device selection.
- Adjust `wake_word_sensitivity` in config.

**Audio Playback Issues**
- Ensure speaker configuration and sample rates match.

---

## Development

```bash
# Unit tests
pytest tests/

# Format code
black wallie_voice_bot.py workers/

# Lint
ruff check .

# Type checking
mypy wallie_voice_bot.py --strict
```

---

## Privacy & Security

- **No Network Access**: Blocks outbound connections (except optional metrics)
- **Audio Retention**: Mic audio kept ≤10 min in RAM only
- **Transcript Privacy**: Deleted after 24h unless archiving enabled
- **Process Isolation**: Each component sandboxed
- **Secure Defaults**: Minimal permissions, no external dependencies

---

## License

MIT License – See LICENSE file

---

## Acknowledgments

Built with:
- [Picovoice Porcupine](https://picovoice.ai/) – Wake word detection
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper) – Speech recognition  
- [vLLM](https://vllm.ai/) – Fast LLM inference
- [Coqui TTS](https://github.com/coqui-ai/TTS) – Neural speech synthesis

---

## Support

- Issues: [GitHub Issues](https://github.com/yourusername/wallie-voice-bot/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/wallie-voice-bot/discussions)
- Wiki: [Documentation Wiki](https://github.com/yourusername/wallie-voice-bot/wiki)