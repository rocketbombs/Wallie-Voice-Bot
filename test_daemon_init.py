#!/usr/bin/env python3
"""
Test daemon initialization without full startup
"""

import asyncio
import sys
from pathlib import Path

async def test_daemon_init():
    """Test daemon initialization"""
    print("Testing daemon initialization...")
    
    try:
        from wallie_voice_bot import WallieDaemon
        
        # Create a temporary config
        config_path = Path.home() / ".wallie_voice_bot" / "config.toml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not config_path.exists():
            # Create minimal config
            config_content = """
wake_word = "wallie"
wake_word_sensitivity = 0.7
asr_model = "tiny.en"
asr_device = "cuda"
"""
            with open(config_path, "w") as f:
                f.write(config_content)
            print(f"✓ Created config at {config_path}")
        
        # Initialize daemon (but don't run)
        daemon = WallieDaemon(config_path)
        print("✓ Daemon initialized successfully")
        print(f"  - Wake word: {daemon.config.wake_word}")
        print(f"  - ASR device: {daemon.config.asr_device}")
        print(f"  - Workers configured: {len(daemon.workers)} process slots")
        print(f"  - Queues configured: {len(daemon.queues)} queues")
        
        # Test configuration loading
        config2 = daemon._load_config()
        print(f"✓ Configuration reloads successfully")
        
        # Test logging setup  
        logger = daemon.logger
        logger.info("Test log message")
        print("✓ Logging system works")
        
        return True
        
    except Exception as e:
        print(f"✗ Daemon initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_daemon_init())
    sys.exit(0 if success else 1)
