#!/usr/bin/env python3
"""
Debug queue communication issues in Wallie Voice Bot
"""

import multiprocessing as mp
import time
import numpy as np
import os

def test_basic_queue():
    """Test basic queue operations with detailed debugging"""
    print("Testing Basic Queue Operations")
    print("-" * 40)
    
    # Set environment variable to avoid OpenMP conflicts
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    try:
        # Create queue
        test_queue = mp.Queue()
        print("✓ Queue created successfully")
        
        # Test putting messages
        test_messages = [
            {'type': 'wake_word_detected', 'timestamp': time.time()},
            {'type': 'audio_chunk', 'data': np.random.random(10).tolist()},  # Smaller data
            {'type': 'transcription', 'text': 'hello world'},
            {'type': 'response', 'text': 'Hello! How can I help you?'},
            {'type': 'audio_generated', 'audio_data': np.random.random(10).tolist()}
        ]
        
        print(f"Putting {len(test_messages)} messages...")
        for i, msg in enumerate(test_messages):
            test_queue.put(msg)
            print(f"  Put message {i+1}: {msg['type']}")
        
        # Check queue size
        print(f"Queue size after putting: {test_queue.qsize()}")
        
        # Get messages with timeout
        received = []
        print("Getting messages...")
        for i in range(len(test_messages)):
            try:
                msg = test_queue.get(timeout=1)
                received.append(msg)
                print(f"  Got message {i+1}: {msg['type']}")
            except:
                print(f"  Timeout getting message {i+1}")
                break
        
        print(f"Final queue size: {test_queue.qsize()}")
        print(f"Sent: {len(test_messages)}, Received: {len(received)}")
        
        return len(received) == len(test_messages)
        
    except Exception as e:
        print(f"✗ Queue test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiprocess_queue():
    """Test queue communication between processes"""
    print("\nTesting Multi-Process Queue")
    print("-" * 40)
    
    def producer(queue, messages):
        """Producer process"""
        for msg in messages:
            queue.put(msg)
            time.sleep(0.1)  # Small delay
        queue.put(None)  # Sentinel value
    
    def consumer(queue, result_list):
        """Consumer process"""
        while True:
            try:
                msg = queue.get(timeout=5)
                if msg is None:  # Sentinel value
                    break
                result_list.append(msg)
            except:
                break
    
    try:
        # Shared objects
        manager = mp.Manager()
        queue = manager.Queue()
        result_list = manager.list()
        
        test_messages = [
            {'type': 'test1', 'data': 'hello'},
            {'type': 'test2', 'data': 'world'},
            {'type': 'test3', 'data': 'test'}
        ]
        
        # Create processes
        prod_process = mp.Process(target=producer, args=(queue, test_messages))
        cons_process = mp.Process(target=consumer, args=(queue, result_list))
        
        # Start processes
        prod_process.start()
        cons_process.start()
        
        # Wait for completion
        prod_process.join(timeout=10)
        cons_process.join(timeout=10)
        
        print(f"Sent: {len(test_messages)}, Received: {len(result_list)}")
        
        return len(result_list) == len(test_messages)
        
    except Exception as e:
        print(f"✗ Multi-process queue test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Wallie Voice Bot - Queue Communication Debug")
    print("=" * 50)
    
    # Test basic queue
    basic_result = test_basic_queue()
    
    # Test multi-process queue
    mp_result = test_multiprocess_queue()
    
    print("\n" + "=" * 50)
    print("RESULTS:")
    print(f"Basic Queue: {'✓' if basic_result else '✗'}")
    print(f"Multi-Process Queue: {'✓' if mp_result else '✗'}")
    
    if basic_result and mp_result:
        print("✓ Queue communication working properly")
    else:
        print("✗ Queue communication has issues")
