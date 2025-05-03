"""Advanced streaming tests for CLIP model with Redis cache integration.
Multi-GPU support with CUDA streams
Advanced memory management
Performance monitoring and metrics
Optimized for A100 GPUs on GCP
Streaming dataset support
Advanced Redis caching with compression
GPU metrics collection
"""
import pytest
import pytest_asyncio
import numpy as np
import torch
import asyncio
from typing import Dict, List, Tuple, AsyncGenerator, Optional, Any
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from datasets import Dataset, IterableDataset, load_dataset, concatenate_datasets
from caching.redis_cache import RedisEmbeddingCache
from input.dataset_config import DatasetConfig
from tqdm.asyncio import tqdm_asyncio
import logging
from contextlib import asynccontextmanager, nullcontext
import clip
import time
from torch.profiler import profile, record_function, ProfilerActivity
from PIL import Image
import io
from dataclasses import dataclass

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GPUConfig:
    def __init__(self):
        self.num_gpus = torch.cuda.device_count()
        self.devices = [torch.device(f'cuda:{i}') for i in range(self.num_gpus)]
        self.main_device = torch.device('cuda:0' if self.num_gpus > 0 else 'cpu')
        self.streams = {device: torch.cuda.Stream(device=device) 
                       for device in self.devices if str(device).startswith('cuda')}
        
    def optimize_settings(self):
        if self.num_gpus > 0:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    def get_next_device(self, batch_idx):
        return self.devices[batch_idx % self.num_gpus] if self.num_gpus > 0 else torch.device('cpu')

class StreamingMetrics:
    """Track streaming performance metrics with GPU monitoring."""
    def __init__(self, gpu_config: GPUConfig):
        self.processing_times = []
        self.batch_sizes = []
        self.cache_latencies = []
        self.throughput = []
        self._start_time = None
        self.gpu_config = gpu_config
        self.gpu_metrics = []

    def start_batch(self):
        self._start_time = time.perf_counter()
        if self.gpu_config.num_gpus > 0:
            self._record_gpu_metrics()

    def _record_gpu_metrics(self):
        metrics = {}
        for i in range(self.gpu_config.num_gpus):
            metrics[f'gpu_{i}'] = {
                'memory_used': torch.cuda.memory_allocated(i) / 1e9,  # GB
                'utilization': torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else None
            }
        self.gpu_metrics.append(metrics)

    def end_batch(self, batch_size: int, cache_latency: float):
        if self._start_time is None:
            return
        
        processing_time = time.perf_counter() - self._start_time
        self.processing_times.append(processing_time)
        self.batch_sizes.append(batch_size)
        self.cache_latencies.append(cache_latency)
        self.throughput.append(batch_size / processing_time)

    def get_summary(self) -> Dict[str, float]:
        summary = {
            "avg_processing_time": np.mean(self.processing_times),
            "p95_processing_time": np.percentile(self.processing_times, 95),
            "avg_throughput": np.mean(self.throughput),
            "avg_cache_latency": np.mean(self.cache_latencies),
            "total_samples": sum(self.batch_sizes)
        }
        
        if self.gpu_metrics:
            for gpu_id in range(self.gpu_config.num_gpus):
                gpu_memory = [m[f'gpu_{gpu_id}']['memory_used'] for m in self.gpu_metrics]
                summary[f'gpu_{gpu_id}_avg_memory'] = np.mean(gpu_memory)
                summary[f'gpu_{gpu_id}_peak_memory'] = np.max(gpu_memory)
        
        return summary

@pytest.fixture(scope="session")
def gpu_config():
    config = GPUConfig()
    config.optimize_settings()
    return config

@pytest.fixture
async def dataset_config() -> DatasetConfig:
    """Fixture for dataset configuration."""
    return DatasetConfig(
        image_datasets=[
            "laion/laion-art",
            "facebook/image-similarity-challenge"
        ],
        text_datasets=[
            "facebook/flores-200"
        ],
        batch_size=32,
        max_text_length=77,
        image_size=224,
        num_proc=4
    )

@pytest_asyncio.fixture
async def redis_cache() -> AsyncGenerator[RedisEmbeddingCache, None]:
    """Setup Redis cache for testing."""
    config = {
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "password": None,
            "namespace": "test_advanced:",
            "ttl": 3600,
            "max_connections": 20,
            "connection_timeout": 5.0,
            "socket_keepalive": True,
            "socket_timeout": 2.0,
            "retry_on_timeout": True,
            "health_check_interval": 30,
        },
        "max_pipeline_size": 1000,
        "compression": {
            "enabled": True,
            "algorithm": "zstd",
            "level": 3,
            "min_size": 1024,
        },
        "encoding": "utf-8",
        "decode_responses": False,
        "serializer": "msgpack",
        "enable_metrics": True,
        "metrics_namespace": "clip_cache",
        "log_metrics_interval": 60,
    }
    
    cache = RedisEmbeddingCache(config)
    try:
        await cache.redis.flushdb()
        yield cache
    finally:
        await cache.redis.aclose()  # Using aclose() instead of close()

@pytest.fixture(scope="session")
def clip_model(gpu_config):
    """Load CLIP model with multi-GPU support."""
    models = {}
    for device in gpu_config.devices:
        with torch.cuda.device(device):
            model, preprocess = clip.load("ViT-L/14@336px", device=device)
            models[device] = (model, preprocess)
    return models

@pytest.fixture(scope="session")
def clip_processor():
    """CLIP processor with modern settings."""
    return CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14-336",
        torch_dtype=torch.float16
    )

@asynccontextmanager
async def batch_processor(model: torch.nn.Module, device: torch.device, stream: Optional[torch.cuda.Stream] = None):
    """Advanced async context manager for optimized batch processing."""
    try:
        torch.cuda.empty_cache()
        stream_ctx = torch.cuda.stream(stream) if stream else nullcontext()
        with torch.inference_mode(), \
             stream_ctx, \
             torch.amp.autocast(device_type=str(device).split(':')[0], dtype=torch.float16), \
             torch.backends.cudnn.flags(enabled=True, benchmark=True):
            yield
    finally:
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()

async def process_streaming_batch(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    metrics: Optional[StreamingMetrics] = None,
    cache: Optional[RedisEmbeddingCache] = None  # Add cache parameter
) -> Tuple[np.ndarray, np.ndarray]:
    """Process a streaming batch with advanced optimizations and caching."""
    if metrics:
        metrics.start_batch()
    
    async with batch_processor(model, device):
        # Efficient tensor movement
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        
        # Try to get from cache first if cache is provided
        if cache:
            cache_key = f"batch_{hash(str(pixel_values.cpu().numpy().tobytes()))}"
            cached_features = await cache.get(cache_key)
            if cached_features is not None:
                if metrics:
                    metrics.end_batch(len(pixel_values), 0.0)
                return cached_features
        
        # Profile the forward pass
        with record_function("model_inference"):
            with torch.cuda.amp.autocast():
                # Fix batch dimension before passing to model
                if pixel_values.ndim == 3:
                    # If 3D (single image), add batch dimension
                    pixel_values = pixel_values.unsqueeze(0)  # [1, 3, 224, 224]
                
                features = model.encode_image(pixel_values)
                features = torch.nn.functional.normalize(features, p=2, dim=-1)
        
        # Efficient CPU transfer
        features_np = features.cpu().numpy()
        
        # Cache the results if cache is provided
        if cache:
            await cache.set(cache_key, features_np)
        
        if metrics:
            metrics.end_batch(len(pixel_values), 0.0)
        
        return features_np

@pytest.mark.asyncio
async def test_streaming_embedding_generation(
    clip_model: Dict[torch.device, Tuple[torch.nn.Module, Any]],
    clip_processor: CLIPProcessor,
    dataset_config: DatasetConfig,
    gpu_config: GPUConfig
):
    """Test streaming embedding generation with multi-GPU support."""
    metrics = StreamingMetrics(gpu_config)
    
    # Load streaming datasets
    image_dataset = await dataset_config.load_streaming_dataset(mode="image")
    text_dataset = await dataset_config.load_streaming_dataset(mode="text")
    
    dataloader = dataset_config.create_dataloader(
        image_dataset=image_dataset,
        text_dataset=text_dataset
    )

    async for batch in dataloader:
        device = gpu_config.get_next_device(batch['batch_idx'])
        model, _ = clip_model[device]
        
        metrics.start_batch()
        async with batch_processor(model, device, gpu_config.streams.get(device)):
            # Process batch using dataset_config's processing methods
            processed_batch = dataset_config.process_batch(
                batch,
                clip_processor,
                device
            )
            
            image_embeddings = model.get_image_features(**processed_batch['image'])
            text_embeddings = model.get_text_features(**processed_batch['text'])
            
            # Verify embeddings
            assert image_embeddings.shape[1] == text_embeddings.shape[1] == model.config.projection_dim
            assert torch.isfinite(image_embeddings).all()
            assert torch.isfinite(text_embeddings).all()
            
        metrics.end_batch(len(batch['image']), 0.0)
        
        if metrics.get_summary()['total_samples'] >= dataset_config.max_samples:
            break
    
    summary = metrics.get_summary()
    logger.info(f"Streaming embedding generation metrics: {summary}")
    assert summary['avg_throughput'] > 0

@pytest.mark.asyncio
async def test_streaming_cache_operations(
    clip_model: Dict[torch.device, Tuple[torch.nn.Module, Any]],
    clip_processor: CLIPProcessor,
    dataset_config: DatasetConfig,
    redis_cache: RedisEmbeddingCache,
    gpu_config: GPUConfig
):
    """Test streaming cache operations with Redis."""
    metrics = StreamingMetrics(gpu_config)
    
    # Load streaming datasets
    image_dataset = await dataset_config.load_streaming_dataset(mode="image")
    text_dataset = await dataset_config.load_streaming_dataset(mode="text")
    
    dataloader = dataset_config.create_dataloader(
        image_dataset=image_dataset,
        text_dataset=text_dataset
    )

    async for batch in dataloader:
        device = gpu_config.get_next_device(batch['batch_idx'])
        model, _ = clip_model[device]
        
        metrics.start_batch()
        async with batch_processor(model, device, gpu_config.streams.get(device)):
            processed_batch = dataset_config.process_batch(
                batch,
                clip_processor,
                device
            )
            
            # Generate and cache embeddings
            cache_start = time.perf_counter()
            
            # Try to get from cache first
            cache_keys = [f"img_{idx}" for idx in batch['image_ids']]
            cached_embeddings = await redis_cache.mget(cache_keys)
            
            # Generate embeddings for cache misses
            miss_indices = [i for i, emb in enumerate(cached_embeddings) if emb is None]
            if miss_indices:
                miss_images = {k: processed_batch['image'][k][miss_indices] 
                             for k in processed_batch['image'].keys()}
                new_embeddings = model.get_image_features(**miss_images)
                
                # Cache the new embeddings
                miss_keys = [cache_keys[i] for i in miss_indices]
                await redis_cache.mset(dict(zip(miss_keys, new_embeddings.cpu().numpy())))
            
            cache_latency = time.perf_counter() - cache_start
            metrics.end_batch(len(batch['image']), cache_latency)
        
        if metrics.get_summary()['total_samples'] >= dataset_config.max_samples:
            break
    
    summary = metrics.get_summary()
    logger.info(f"Streaming cache operation metrics: {summary}")
    assert summary['avg_cache_latency'] < 1.0  # Cache operations should be fast

@pytest.mark.asyncio
async def test_streaming_similarity_search(
    clip_model: Dict[torch.device, Tuple[torch.nn.Module, Any]],
    clip_processor: CLIPProcessor,
    dataset_config: DatasetConfig,
    gpu_config: GPUConfig
):
    """Test similarity search with GPU acceleration."""
    metrics = StreamingMetrics(gpu_config)
    
    # Load streaming datasets
    image_dataset = await dataset_config.load_streaming_dataset(mode="image")
    text_dataset = await dataset_config.load_streaming_dataset(mode="text")
    
    dataloader = dataset_config.create_dataloader(
        image_dataset=image_dataset,
        text_dataset=text_dataset
    )

    # Build search index
    index_embeddings = []
    index_ids = []
    
    async for batch in dataloader:
        device = gpu_config.get_next_device(batch['batch_idx'])
        model, _ = clip_model[device]
        
        metrics.start_batch()
        async with batch_processor(model, device, gpu_config.streams.get(device)):
            processed_batch = dataset_config.process_batch(
                batch,
                clip_processor,
                device
            )
            
            # Generate embeddings for both modalities
            image_embeddings = model.get_image_features(**processed_batch['image'])
            text_embeddings = model.get_text_features(**processed_batch['text'])
            
            # Store embeddings and IDs for index
            index_embeddings.append(image_embeddings.cpu())
            index_ids.extend(batch['image_ids'])
            
        metrics.end_batch(len(batch['image']), 0.0)
        
        if metrics.get_summary()['total_samples'] >= dataset_config.max_samples:
            break
    
    # Combine embeddings
    index_embeddings = torch.cat(index_embeddings, dim=0)
    
    # Perform similarity search
    device = gpu_config.main_device
    query_idx = 0
    query = index_embeddings[query_idx].to(device)
    
    # Compute similarities efficiently on GPU
    chunk_size = dataset_config.batch_size
    similarities = []
    
    for i in range(0, len(index_embeddings), chunk_size):
        chunk = index_embeddings[i:i + chunk_size].to(device)
        chunk_similarities = torch.matmul(chunk, query)
        similarities.append(chunk_similarities.cpu())
    
    similarities = torch.cat(similarities)
    
    # Validate results
    assert len(similarities) == len(index_embeddings)
    assert torch.all((-1 <= similarities) & (similarities <= 1))
    assert torch.argmax(similarities).item() == query_idx  # Self should be most similar
    
    # Get top-k results
    k = 5
    top_k_values, top_k_indices = torch.topk(similarities, k)
    top_k_ids = [index_ids[i] for i in top_k_indices.tolist()]
    
    logger.info(f"Top {k} similar items: {list(zip(top_k_ids, top_k_values.tolist()))}")
    
    # Log performance
    summary = metrics.get_summary()
    logger.info(f"Search metrics: {summary}")
    assert summary['avg_throughput'] > 0

@pytest.mark.asyncio
async def test_streaming_performance(
    clip_model: Dict[torch.device, Tuple[torch.nn.Module, Any]],
    clip_processor: CLIPProcessor,
    dataset_config: DatasetConfig,
    redis_cache: RedisEmbeddingCache,
    gpu_config: GPUConfig
):
    """Test streaming performance with multi-GPU support."""
    metrics = StreamingMetrics(gpu_config)
    
    # Load streaming datasets
    image_dataset = await dataset_config.load_streaming_dataset(mode="image")
    text_dataset = await dataset_config.load_streaming_dataset(mode="text")
    
    dataloader = dataset_config.create_dataloader(
        image_dataset=image_dataset,
        text_dataset=text_dataset
    )

    # Track performance metrics
    total_samples = 0
    total_cache_hits = 0
    total_cache_misses = 0
    batch_times = []
    
    async for batch in dataloader:
        device = gpu_config.get_next_device(batch['batch_idx'])
        model, _ = clip_model[device]
        
        batch_start = time.perf_counter()
        metrics.start_batch()
        
        async with batch_processor(model, device, gpu_config.streams.get(device)):
            processed_batch = dataset_config.process_batch(
                batch,
                clip_processor,
                device
            )
            
            # Try cache first
            cache_keys = [f"img_{idx}" for idx in batch['image_ids']]
            cached_embeddings = await redis_cache.mget(cache_keys)
            
            # Count cache hits/misses
            miss_indices = [i for i, emb in enumerate(cached_embeddings) if emb is None]
            total_cache_hits += len(cached_embeddings) - len(miss_indices)
            total_cache_misses += len(miss_indices)
            
            # Generate embeddings for cache misses
            if miss_indices:
                miss_images = {k: processed_batch['image'][k][miss_indices] 
                             for k in processed_batch['image'].keys()}
                new_embeddings = model.get_image_features(**miss_images)
                
                # Cache new embeddings
                miss_keys = [cache_keys[i] for i in miss_indices]
                await redis_cache.mset(dict(zip(miss_keys, new_embeddings.cpu().numpy())))
            
            # Process text embeddings
            text_embeddings = model.get_text_features(**processed_batch['text'])
            
        batch_time = time.perf_counter() - batch_start
        batch_times.append(batch_time)
        metrics.end_batch(len(batch['image']), batch_time)
        
        total_samples += len(batch['image'])
        if total_samples >= dataset_config.max_samples:
            break
    
    # Calculate performance metrics
    avg_batch_time = sum(batch_times) / len(batch_times)
    throughput = total_samples / sum(batch_times)
    cache_hit_rate = total_cache_hits / (total_cache_hits + total_cache_misses) if total_cache_hits + total_cache_misses > 0 else 0
    
    # Log detailed metrics
    summary = metrics.get_summary()
    logger.info(f"Performance metrics:")
    logger.info(f"- Average batch time: {avg_batch_time:.3f}s")
    logger.info(f"- Throughput: {throughput:.2f} samples/s")
    logger.info(f"- Cache hit rate: {cache_hit_rate:.2%}")
    logger.info(f"- GPU metrics: {summary}")
    
    # Assertions for performance requirements
    assert throughput > 10, f"Throughput {throughput:.2f} samples/s is below minimum requirement"
    assert avg_batch_time < 1.0, f"Average batch time {avg_batch_time:.3f}s exceeds maximum allowed"
    assert cache_hit_rate > 0.1, f"Cache hit rate {cache_hit_rate:.2%} is below minimum requirement"

    """
    Modern Model Architecture:
Uses CLIP-large (ViT-L/14@336px) instead of base model
Automatic device mapping for multi-GPU support
FP16 precision for better performance
Memory Optimizations:
Automatic mixed precision (AMP) training
Efficient memory management with torch.cuda.empty_cache()
Non-blocking tensor transfers
Optimized CUDA settings
Streaming Capabilities:
True streaming dataset support using HuggingFace's IterableDataset
Efficient batch processing with async generators
Memory-efficient data loading
Performance Features:
Connection pooling for Redis
Larger pipeline sizes for better throughput
Concurrent stream processing
Detailed performance metrics and logging
Modern Testing Approaches:
Async context managers for resource management
Progress bars with tqdm_asyncio
Comprehensive performance testing
Proper cleanup and resource handling
    """