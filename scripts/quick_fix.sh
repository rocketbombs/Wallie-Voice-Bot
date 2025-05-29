#!/bin/bash
# Quick fix for worker startup issues

echo "🔧 Applying startup fixes..."

# Ensure PV_ACCESS_KEY is exported
export PV_ACCESS_KEY="${PV_ACCESS_KEY}"

# Create minimal test config
cat > ~/.wallie_voice_bot/config.toml << EOF
wake_word = "picovoice"  # Use built-in word
wake_word_sensitivity = 0.5

asr_model = "tiny.en"
asr_device = "cpu"  # Force CPU for testing
asr_compute_type = "int8"

llm_model = "gpt2"  # Minimal model
llm_max_tokens = 128
llm_temperature = 0.7
llm_gpu_memory_fraction = 0.3

tts_engine = "pyttsx3"  # Offline TTS
tts_voice = "0"
tts_language = "en"

watchdog_interval_sec = 5
enable_prometheus = false
archive_transcripts = false
EOF

# Test minimal startup
echo "Testing minimal configuration..."
python -c "
import os
print(f'PV_ACCESS_KEY: {\"Set\" if os.environ.get(\"PV_ACCESS_KEY\") else \"NOT SET\"}')

# Test critical imports
try:
    import pvporcupine
    print('✓ Porcupine OK')
except Exception as e:
    print(f'✗ Porcupine: {e}')

try:
    import faster_whisper
    print('✓ Whisper OK')
except Exception as e:
    print(f'✗ Whisper: {e}')
"

echo ""
echo "Starting Wallie with fixes..."
python wallie_voice_bot.py
