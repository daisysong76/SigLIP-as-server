"""Advanced tests for CLIP model functionality with Redis cache integration."""
import pytest
import pytest_asyncio
import numpy as np
import torch
from PIL import Image
import io
import asyncio
from typing import Dict, List, Tuple
from pathlib import Path
import clip
from transformers import CLIPProcessor
from caching.redis_cache import create_clip_cache
from input.dataset_config import DatasetConfig

@pytest.fixture(scope="session")
def dataset_config():
    """Dataset configuration fixture."""
    return DatasetConfig()

@pytest.fixture(scope="session")
def clip_processor():
    """CLIP processor fixture."""
    return CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@pytest.fixture(scope="session")
def device():
    """PyTorch device fixture."""
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture(scope="session")
def clip_model(device):
    """CLIP model fixture."""
    model, _ = clip.load("ViT-B/32", device=device)
    return model

@pytest_asyncio.fixture
async def redis_cache():
    """Redis cache fixture."""
    config = {
        "redis": {
            "host": "localhost",
            "port": 6379,
            "password": None,
            "db": 0,
            "ttl": 3600,
            "namespace": "test_clip:",
            "compression": True
        }
    }
    cache = await create_clip_cache(config)
    yield cache
    await cache.flush_namespace()
    await cache.close()

@pytest.fixture(scope="session")
def test_dataset(dataset_config, clip_processor):
    """Load test dataset."""
    return dataset_config.load_streaming_dataset(
        processor=clip_processor,
        split="train",
        streaming=False  # For testing, we want a fixed dataset
    )

@pytest.fixture(scope="session")
def test_dataloader(dataset_config, test_dataset):
    """Create test dataloader."""
    return dataset_config.create_dataloader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=True
    )

async def generate_embeddings(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate embeddings for a batch of inputs."""
    with torch.no_grad():
        # Move inputs to device
        pixel_values = batch["pixel_values"].to(device)
        text_features = batch["text_features"].to(device)
        
        # Generate embeddings
        image_features = model.encode_image(pixel_values)
        text_features = model.encode_text(text_features)
        
        # Normalize embeddings
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    return text_features.cpu().numpy(), image_features.cpu().numpy()

@pytest.mark.asyncio
async def test_clip_embedding_generation(clip_model, test_dataloader, device):
    """Test CLIP embedding generation with real data."""
    # Get first batch
    batch = next(iter(test_dataloader))
    
    # Generate embeddings
    text_embeddings, image_embeddings = await generate_embeddings(
        clip_model, batch, device
    )
    
    # Check shapes
    assert text_embeddings.shape[1] == 512
    assert image_embeddings.shape[1] == 512
    assert len(text_embeddings) == len(image_embeddings) == batch["pixel_values"].size(0)
    
    # Check normalization
    np.testing.assert_array_almost_equal(
        np.linalg.norm(text_embeddings, axis=1),
        np.ones(len(text_embeddings)),
        decimal=6
    )
    np.testing.assert_array_almost_equal(
        np.linalg.norm(image_embeddings, axis=1),
        np.ones(len(image_embeddings)),
        decimal=6
    )

@pytest.mark.asyncio
async def test_redis_cache_with_clip_embeddings(clip_model, test_dataloader, redis_cache, device):
    """Test Redis cache with real CLIP embeddings."""
    # Get first batch
    batch = next(iter(test_dataloader))
    
    # Generate embeddings
    text_embeddings, image_embeddings = await generate_embeddings(
        clip_model, batch, device
    )
    
    # Test storing and retrieving text embeddings
    for i, embedding in enumerate(text_embeddings):
        key = f"text_embedding_{i}"
        await redis_cache.set(key, embedding)
        cached_embedding = await redis_cache.get(key)
        np.testing.assert_array_almost_equal(embedding, cached_embedding)
    
    # Test storing and retrieving image embeddings
    for i, embedding in enumerate(image_embeddings):
        key = f"image_embedding_{i}"
        await redis_cache.set(key, embedding)
        cached_embedding = await redis_cache.get(key)
        np.testing.assert_array_almost_equal(embedding, cached_embedding)

@pytest.mark.asyncio
async def test_similarity_search(clip_model, test_dataloader, redis_cache, device):
    """Test similarity search with real embeddings."""
    # Get first batch
    batch = next(iter(test_dataloader))
    
    # Generate embeddings
    text_embeddings, image_embeddings = await generate_embeddings(
        clip_model, batch, device
    )
    
    # Store embeddings
    for i, embedding in enumerate(text_embeddings):
        await redis_cache.set(f"text_{i}", embedding)
    
    # Test similarity search
    query_embedding = text_embeddings[0]
    results = await redis_cache.find_similar(query_embedding, top_k=5, threshold=0.0)
    
    assert len(results) > 0
    similarities = [sim for _, sim in results]
    assert all(-1 <= sim <= 1 for sim in similarities)
    assert all(similarities[i] >= similarities[i+1] for i in range(len(similarities)-1))

@pytest.mark.asyncio
async def test_batch_operations(clip_model, test_dataloader, redis_cache, device):
    """Test batch operations with real data."""
    # Get multiple batches
    batches = [next(iter(test_dataloader)) for _ in range(3)]
    all_embeddings = []
    
    # Generate embeddings for all batches
    for batch in batches:
        text_embeddings, _ = await generate_embeddings(clip_model, batch, device)
        all_embeddings.extend(text_embeddings)
    
    # Test batch write
    items = {f"batch_emb_{i}": emb for i, emb in enumerate(all_embeddings)}
    await redis_cache.set_many(items)
    
    # Test batch read
    results = await redis_cache.get_many(list(items.keys()))
    assert len(results) == len(items)
    
    # Verify results
    for key, embedding in items.items():
        np.testing.assert_array_almost_equal(embedding, results[key])

@pytest.mark.asyncio
async def test_error_handling(redis_cache):
    """Test error handling with invalid inputs."""
    # Test invalid key
    assert await redis_cache.get("nonexistent_key") is None
    
    # Test invalid embedding format
    with pytest.raises(TypeError):
        await redis_cache.set("invalid_embedding", "not_an_array")
    
    # Test invalid similarity search parameters
    with pytest.raises(ValueError):
        await redis_cache.find_similar(np.zeros(512), top_k=0)
    with pytest.raises(ValueError):
        await redis_cache.find_similar(np.zeros(512), threshold=2.0)

@pytest.mark.asyncio
async def test_concurrent_operations(clip_model, test_dataloader, redis_cache, device):
    """Test concurrent cache operations with real data."""
    # Get first batch
    batch = next(iter(test_dataloader))
    text_embeddings, _ = await generate_embeddings(clip_model, batch, device)
    
    # Test concurrent writes
    async def write_embedding(i):
        await redis_cache.set(f"concurrent_write_{i}", text_embeddings[0])
    
    await asyncio.gather(*[write_embedding(i) for i in range(10)])
    
    # Test concurrent reads
    async def read_embedding(i):
        return await redis_cache.get(f"concurrent_write_{i}")
    
    results = await asyncio.gather(*[read_embedding(i) for i in range(10)])
    assert all(np.array_equal(r, text_embeddings[0]) for r in results)

@pytest.mark.asyncio
async def test_cross_modal_similarity(clip_model, test_dataloader, redis_cache, device):
    """Test cross-modal (text-to-image) similarity."""
    # Get first batch
    batch = next(iter(test_dataloader))
    
    # Generate embeddings
    text_embeddings, image_embeddings = await generate_embeddings(
        clip_model, batch, device
    )
    
    # Store image embeddings
    for i, embedding in enumerate(image_embeddings):
        await redis_cache.set(f"image_{i}", embedding)
    
    # Use text embedding to search for similar images
    query_embedding = text_embeddings[0]
    results = await redis_cache.find_similar(query_embedding, top_k=5, threshold=0.0)
    
    assert len(results) > 0
    # Check that all results are image keys
    assert all(r[0].startswith("image_") for r in results)
    # Check that similarities are between -1 and 1
    assert all(-1 <= r[1] <= 1 for r in results)

@pytest.mark.asyncio
async def test_cache_performance_with_clip(clip_model, test_dataloader, redis_cache, device):
    """Test cache performance with CLIP embeddings."""
    # Get first batch
    batch = next(iter(test_dataloader))
    
    # Generate embeddings
    text_embeddings, _ = await generate_embeddings(
        clip_model, batch, device
    )
    
    # Measure batch write performance
    start_time = asyncio.get_event_loop().time()
    items = {f"perf_text_{i}": emb for i, emb in enumerate(text_embeddings)}
    await redis_cache.set_many(items)
    write_time = asyncio.get_event_loop().time() - start_time
    
    # Measure batch read performance
    start_time = asyncio.get_event_loop().time()
    results = await redis_cache.get_many(list(items.keys()))
    read_time = asyncio.get_event_loop().time() - start_time
    
    # Check performance metrics
    assert write_time / len(text_embeddings) < 0.01  # Less than 10ms per embedding
    assert read_time / len(text_embeddings) < 0.01
    assert len(results) == len(items)
    
    # Get cache stats
    stats = await redis_cache.get_stats()
    assert stats["hit_rate"] > 0.95  # Should have high hit rate 

@pytest.mark.asyncio
async def test_large_batch_operations(clip_model, test_dataloader, redis_cache, device):
    """Test operations with large batches of embeddings."""
    # Get first batch
    batch = next(iter(test_dataloader))
    
    # Generate a large number of embeddings
    large_texts = batch["text_features"] * 100  # 500 embeddings
    text_embeddings, _ = await generate_embeddings(
        clip_model, batch, device
    )
    
    # Test large batch write
    large_dict = {f"large_batch_{i}": emb for i, emb in enumerate(text_embeddings)}
    await redis_cache.set_many(large_dict)
    
    # Test large batch read
    results = await redis_cache.get_many(list(large_dict.keys()))
    assert len(results) == len(large_dict)
    
    # Test large batch similarity search
    query = text_embeddings[0]
    # Ensure query is normalized
    query = query / np.linalg.norm(query)
    results = await redis_cache.find_similar(query, top_k=100, threshold=0.0)
    assert len(results) > 0
    assert all(-1 <= sim <= 1 for _, sim in results)
    
    # Check that we have high similarity scores for identical embeddings
    high_similarity_count = sum(1 for _, sim in results if sim > 0.99)
    assert high_similarity_count > 0, "Should find at least one highly similar embedding"
    
    # Check that results are sorted by similarity (descending)
    similarities = [sim for _, sim in results]
    assert all(similarities[i] >= similarities[i+1] for i in range(len(similarities)-1))

@pytest.mark.asyncio
async def test_compression_effectiveness(redis_cache):
    """Test the effectiveness of embedding compression."""
    # Get first batch
    batch = next(iter(test_dataloader))
    
    # Generate random embedding
    embedding = batch["text_features"][0].astype(np.float32)
    embedding /= np.linalg.norm(embedding)
    
    # Store with and without compression
    config_no_compression = {
        "redis": {
            "host": "localhost",
            "port": 6379,
            "password": None,
            "db": 0,
            "ttl": 3600,
            "namespace": "test_clip:",
            "compression": False
        }
    }
    uncompressed_cache = await create_clip_cache(config_no_compression)
    
    await redis_cache.set("compressed_embedding", embedding)
    await uncompressed_cache.set("uncompressed_embedding", embedding)
    
    # Compare retrieved embeddings
    compressed = await redis_cache.get("compressed_embedding")
    uncompressed = await uncompressed_cache.get("uncompressed_embedding")
    
    np.testing.assert_array_almost_equal(compressed, uncompressed, decimal=6)
    await uncompressed_cache.close() 