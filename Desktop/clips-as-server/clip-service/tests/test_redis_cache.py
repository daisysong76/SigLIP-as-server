import pytest
import pytest_asyncio
import numpy as np
from caching.redis_cache import create_clip_cache

@pytest.fixture
def redis_config():
    return {
        "redis": {
            "host": "localhost",
            "port": 6379,
            "password": None,
            "db": 0,
            "ttl": 3600,
            "namespace": "test:",
            "compression": False
        }
    }

@pytest_asyncio.fixture
async def cache(redis_config):
    cache_instance = await create_clip_cache(redis_config)
    yield cache_instance
    await cache_instance.flush_namespace()
    await cache_instance.close()

@pytest.mark.asyncio
async def test_basic_set_get(cache):
    # Test basic set/get operations
    key = "test_key"
    embedding = np.random.rand(512).astype(np.float32)
    
    await cache.set(key, embedding)
    result = await cache.get(key)
    
    assert result is not None
    np.testing.assert_array_almost_equal(embedding, result)

@pytest.mark.asyncio
async def test_batch_operations(cache):
    # Test batch set/get operations
    embeddings = {
        f"key_{i}": np.random.rand(512).astype(np.float32)
        for i in range(5)
    }
    
    await cache.set_many(embeddings)
    results = await cache.get_many(list(embeddings.keys()))
    
    assert len(results) == len(embeddings)
    for key, embedding in embeddings.items():
        np.testing.assert_array_almost_equal(embedding, results[key])

@pytest.mark.asyncio
async def test_cache_stats(cache):
    # Test statistics tracking
    key = "stats_test"
    embedding = np.random.rand(512).astype(np.float32)
    
    await cache.set(key, embedding)
    _ = await cache.get(key)  # Hit
    _ = await cache.get("nonexistent")  # Miss
    
    stats = await cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5

@pytest.mark.asyncio
async def test_similarity_search(cache):
    # Test similarity search functionality
    embeddings = {
        f"key_{i}": np.random.rand(512).astype(np.float32)
        for i in range(10)
    }
    
    # Normalize and store embeddings
    for key, emb in embeddings.items():
        embeddings[key] = emb / np.linalg.norm(emb)
        await cache.set(key, embeddings[key])
    
    query = np.random.rand(512).astype(np.float32)
    query = query / np.linalg.norm(query)
    
    results = await cache.find_similar(query, top_k=5, threshold=0.0)
    assert len(results) > 0
    assert all(0 <= score <= 1 for _, score in results) 