"""Visualize Redis cache benchmark results."""
import asyncio
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from caching.redis_cache import create_clip_cache

async def collect_benchmark_data(num_ops: int = 1000) -> Dict[str, List[float]]:
    """Collect benchmark data for visualization."""
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
    embedding = np.random.rand(512).astype(np.float32)
    
    try:
        # Collect latency data
        set_times = []
        get_times = []
        hit_rates = []
        throughput_data = []
        
        print("Collecting benchmark data...")
        
        # Measure latencies and hit rates
        for i in range(num_ops):
            key = f"bench_key_{i}"
            
            # SET latency
            start = time.perf_counter()
            await cache.set(key, embedding)
            set_times.append((time.perf_counter() - start) * 1000)  # ms
            
            # GET latency
            start = time.perf_counter()
            await cache.get(key)
            get_times.append((time.perf_counter() - start) * 1000)  # ms
            
            # Hit rate
            stats = await cache.get_stats()
            hit_rates.append(stats["hit_rate"])
            
            # Throughput measurement (ops/sec over 100ms windows)
            if i % 100 == 0:
                start = time.perf_counter()
                for _ in range(100):
                    key = f"throughput_key_{_}"
                    await cache.set(key, embedding)
                    await cache.get(key)
                duration = time.perf_counter() - start
                throughput_data.append(200 / duration)  # 200 ops (100 sets + 100 gets)
        
        return {
            "set_times": set_times,
            "get_times": get_times,
            "hit_rates": hit_rates,
            "throughput": throughput_data
        }
    
    finally:
        await cache.flush_namespace()
        await cache.close()

def plot_benchmark_results(data: Dict[str, List[float]]):
    """Create visualizations of benchmark results."""
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 10))
    
    # 1. Latency Distribution
    plt.subplot(2, 2, 1)
    sns.histplot(data=data["set_times"], label="SET", alpha=0.5)
    sns.histplot(data=data["get_times"], label="GET", alpha=0.5)
    plt.title("Operation Latency Distribution")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    plt.legend()
    
    # 2. Hit Rate Over Time
    plt.subplot(2, 2, 2)
    plt.plot(data["hit_rates"])
    plt.title("Cache Hit Rate Over Time")
    plt.xlabel("Operation Number")
    plt.ylabel("Hit Rate")
    
    # 3. Throughput Over Time
    plt.subplot(2, 2, 3)
    plt.plot(data["throughput"])
    plt.title("Throughput Over Time")
    plt.xlabel("Sample Window (100 ops)")
    plt.ylabel("Operations/Second")
    
    # 4. Latency Box Plot
    plt.subplot(2, 2, 4)
    plt.boxplot([data["set_times"], data["get_times"]], labels=["SET", "GET"])
    plt.title("Operation Latency Box Plot")
    plt.ylabel("Latency (ms)")
    
    plt.tight_layout()
    plt.savefig("benchmark_results.png")
    print("\nBenchmark visualization saved as 'benchmark_results.png'")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("\nSET Latency (ms):")
    print(f"  Average: {np.mean(data['set_times']):.2f}")
    print(f"  P95: {np.percentile(data['set_times'], 95):.2f}")
    print(f"  P99: {np.percentile(data['set_times'], 99):.2f}")
    
    print("\nGET Latency (ms):")
    print(f"  Average: {np.mean(data['get_times']):.2f}")
    print(f"  P95: {np.percentile(data['get_times'], 95):.2f}")
    print(f"  P99: {np.percentile(data['get_times'], 99):.2f}")
    
    print(f"\nAverage Throughput: {np.mean(data['throughput']):.2f} ops/second")
    print(f"Final Hit Rate: {data['hit_rates'][-1]:.2%}")

async def main():
    data = await collect_benchmark_data()
    plot_benchmark_results(data)

if __name__ == "__main__":
    asyncio.run(main()) 