"""Basic functionality tests for CLIP service.
This test script will:
Test the Memory Graph:
Add a test memory
Retrieve it
Verify the content
Test the Redis Embedding Cache:
Create a simulated CLIP embedding
Store it in Redis
Retrieve it
Verify it matches the original
"""
import asyncio
from memory.memory_graph import MemoryGraph
from caching.redis_cache import RedisEmbeddingCache
import numpy as np

async def test_basic_functionality():
    # 1. Test Memory Graph
    print("\n=== Testing Memory Graph ===")
    memory = MemoryGraph()
    
    # Test adding and retrieving memories
    test_data = "Test memory content"
    memory.add_memory("test_user", test_data)
    results = memory.get_memory("test_user")
    print(f"Memory Test - Added and Retrieved: {results}")

    # 2. Test Redis Embedding Cache
    print("\n=== Testing Redis Cache ===")
    config = {
        "redis": {
            "host": "localhost",
            "port": 6379,
            "password": None,
            "db": 0,
            "ttl": 3600
        }
    }
    cache = RedisEmbeddingCache(config)
    
    # Test storing and retrieving embeddings
    test_embedding = np.random.rand(512).astype(np.float32)  # Simulated CLIP embedding
    test_key = "test_image_1"
    
    await cache.set(test_key, test_embedding)
    retrieved_embedding = await cache.get(test_key)
    
    if retrieved_embedding is not None:
        match = np.allclose(test_embedding, retrieved_embedding)
        print(f"Cache Test - Embeddings match: {match}")
    else:
        print("Cache Test - Failed to retrieve embedding")

if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_basic_functionality()) 