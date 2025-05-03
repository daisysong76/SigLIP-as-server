"""Simple benchmark script for Redis cache performance testing."""
import asyncio
import time
import numpy as np
from typing import Dict, Any
from caching.redis_cache import create_clip_cache

async def benchmark_basic_operations(cache, num_ops: int = 1000):
    """Benchmark basic set/get operations."""
    print("\nBenchmarking basic operations...")
    
    # Prepare test data
    embedding = np.random.rand(512).astype(np.float32)
    
    # Measure SET latency
    set_times = []
    for i in range(num_ops):
        key = f"bench_key_{i}"
        start = time.perf_counter()
        await cache.set(key, embedding)
        set_times.append(time.perf_counter() - start)
    
    avg_set_time = np.mean(set_times) * 1000  # Convert to ms
    p95_set_time = np.percentile(set_times, 95) * 1000
    p99_set_time = np.percentile(set_times, 99) * 1000
    
    print(f"SET Operation Latency:")
    print(f"  Average: {avg_set_time:.2f}ms")
    print(f"  P95: {p95_set_time:.2f}ms")
    print(f"  P99: {p99_set_time:.2f}ms")
    
    # Measure GET latency
    get_times = []
    for i in range(num_ops):
        key = f"bench_key_{i}"
        start = time.perf_counter()
        await cache.get(key)
        get_times.append(time.perf_counter() - start)
    
    avg_get_time = np.mean(get_times) * 1000
    p95_get_time = np.percentile(get_times, 95) * 1000
    p99_get_time = np.percentile(get_times, 99) * 1000
    
    print(f"\nGET Operation Latency:")
    print(f"  Average: {avg_get_time:.2f}ms")
    print(f"  P95: {p95_get_time:.2f}ms")
    print(f"  P99: {p99_get_time:.2f}ms")
    
    # Measure throughput
    print("\nMeasuring throughput...")
    duration = 5  # seconds
    start_time = time.perf_counter()
    ops = 0
    
    while time.perf_counter() - start_time < duration:
        key = f"throughput_key_{ops}"
        await cache.set(key, embedding)
        _ = await cache.get(key)
        ops += 1
    
    throughput = ops / duration
    print(f"Throughput: {throughput:.2f} ops/second")
    
    # Get cache stats
    stats = await cache.get_stats()
    print("\nCache Stats:")
    print(f"  Total Hits: {stats['hits']}")
    print(f"  Total Misses: {stats['misses']}")
    print(f"  Hit Rate: {stats['hit_rate']:.2%}")
    print(f"  Memory Usage: {stats['memory_usage']}")

async def main():
    config = {
        "redis": {
            "host": "localhost",
            "port": 6379,
            "password": None,
            "db": 0,
            "ttl": 3600,
            "namespace": "benchmark:",
            "compression": False
        }
    }
    
    cache = await create_clip_cache(config)
    
    try:
        await benchmark_basic_operations(cache)
    finally:
        await cache.flush_namespace()
        await cache.close()

if __name__ == "__main__":
    asyncio.run(main()) 