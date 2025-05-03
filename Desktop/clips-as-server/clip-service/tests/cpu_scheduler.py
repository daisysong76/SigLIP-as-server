"""CPU scheduler for dynamic batch size and performance optimization.
cpu_scheduler.py:
Dynamic batch size adjustment based on CPU/memory load
System metrics monitoring
Integration with torch.compile(), NumExpr, and OpenBLAS
Memory-aware optimization
Performance metrics collection"""

import psutil
import numpy as np
import torch
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import pandas as pd
from pathlib import Path
import time
from contextlib import contextmanager
import numexpr as ne
import os
import ctypes
import platform
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import deque
import perf_metrics
import json

logger = logging.getLogger(__name__)

@dataclass
class CPUStats:
    """Enhanced CPU and memory statistics."""
    cpu_percent: float
    memory_percent: float
    available_memory_gb: float
    load_average: Tuple[float, float, float]
    temperature: Optional[float]
    power_consumption: Optional[float]
    cache_stats: Dict[str, float]  # L1/L2/L3 cache stats
    instruction_mix: Dict[str, float]  # SIMD/scalar/memory instruction mix
    memory_bandwidth: float  # Memory bandwidth utilization
    page_faults: int  # Page faults count
    context_switches: int  # Context switches count

@dataclass
class BatchSizeConfig:
    """Enhanced batch size configuration parameters."""
    min_batch_size: int = 1
    max_batch_size: int = 512
    target_cpu_util: float = 80.0
    target_memory_util: float = 85.0
    adjustment_rate: float = 0.1
    cooldown_seconds: float = 5.0
    prefetch_factor: int = 2  # Number of batches to prefetch
    cache_warmup_iterations: int = 3
    simd_threshold: float = 0.8  # SIMD utilization target
    thread_pool_size: Optional[int] = None

class MemoryPrefetcher:
    """Memory prefetching optimization."""
    
    def __init__(self, prefetch_size: int = 2):
        self.prefetch_size = prefetch_size
        self.prefetch_queue = deque(maxlen=prefetch_size)
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
    
    def prefetch_batch(self, data: np.ndarray):
        """Prefetch data into CPU cache."""
        # Pin memory if using torch tensors
        if isinstance(data, torch.Tensor):
            data = data.pin_memory()
        
        # Ensure data is in page-aligned memory
        aligned_data = np.zeros_like(data, align=True)
        np.copyto(aligned_data, data)
        
        # Touch pages to ensure they're in memory
        _ = aligned_data.sum()
        
        return aligned_data
    
    async def prefetch_async(self, data_loader):
        """Asynchronously prefetch next batches."""
        try:
            future_batches = []
            for i in range(self.prefetch_size):
                batch = next(data_loader)
                future = self.thread_pool.submit(self.prefetch_batch, batch)
                future_batches.append(future)
            
            self.prefetch_queue.extend(future_batches)
        except StopIteration:
            pass
    
    def get_next_batch(self):
        """Get next prefetched batch."""
        if self.prefetch_queue:
            return self.prefetch_queue.popleft().result()
        return None

class CPUOptimizer:
    """CPU-specific optimizations."""
    
    def __init__(self):
        self.numa_nodes = self._detect_numa_nodes()
        self.simd_features = self._detect_simd_features()
        self.cache_sizes = self._get_cache_sizes()
    
    def _detect_numa_nodes(self) -> int:
        """Detect number of NUMA nodes."""
        try:
            if platform.system() == "Linux":
                return len([d for d in os.listdir("/sys/devices/system/node/") if d.startswith("node")])
        except:
            pass
        return 1
    
    def _detect_simd_features(self) -> Dict[str, bool]:
        """Detect available SIMD features."""
        features = {
            "AVX512": False,
            "AVX2": False,
            "AVX": False,
            "SSE4.2": False
        }
        
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            flags = info.get('flags', [])
            
            features.update({
                "AVX512": any(f.startswith("avx512") for f in flags),
                "AVX2": "avx2" in flags,
                "AVX": "avx" in flags,
                "SSE4.2": "sse4_2" in flags
            })
        except:
            pass
        
        return features
    
    def _get_cache_sizes(self) -> Dict[str, int]:
        """Get CPU cache sizes."""
        sizes = {
            "L1": 0,
            "L2": 0,
            "L3": 0
        }
        
        try:
            if platform.system() == "Linux":
                for cache_level in [1, 2, 3]:
                    path = f"/sys/devices/system/cpu/cpu0/cache/index{cache_level}/size"
                    if os.path.exists(path):
                        with open(path) as f:
                            size = f.read().strip()
                            sizes[f"L{cache_level}"] = int(size.replace("K", "000"))
        except:
            pass
        
        return sizes
    
    def optimize_memory_access(self, data: np.ndarray) -> np.ndarray:
        """Optimize memory access patterns."""
        # Ensure data is aligned
        if not data.flags['ALIGNED']:
            data = np.asarray(data, align=True)
        
        # Use optimal memory layout
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        
        return data
    
    def enable_simd(self, model: torch.nn.Module):
        """Enable SIMD optimizations for the model."""
        if self.simd_features["AVX512"]:
            torch._C._jit_set_profiling_executor(True)
            torch._C._jit_set_profiling_mode(True)
            torch._C._debug_set_autodiff_subgraph_inlining(False)
        
        return model

class CPUScheduler:
    """Enhanced CPU scheduler with advanced optimizations."""
    
    def __init__(
        self,
        config: BatchSizeConfig,
        metrics_file: str = "cpu_metrics.csv",
        enable_torch_compile: bool = True,
        enable_numexpr: bool = True,
        enable_openblas: bool = True
    ):
        self.config = config
        self.current_batch_size = config.min_batch_size
        self.last_adjustment_time = 0
        self.metrics_file = Path(metrics_file)
        self.metrics_data = []
        
        # Initialize optimizers
        self.cpu_optimizer = CPUOptimizer()
        self.memory_prefetcher = MemoryPrefetcher(config.prefetch_factor)
        
        # Initialize thread pool
        if config.thread_pool_size is None:
            config.thread_pool_size = self.cpu_optimizer.numa_nodes * psutil.cpu_count(logical=False)
        self.thread_pool = ThreadPoolExecutor(max_workers=config.thread_pool_size)
        
        self._setup_optimizations(
            enable_torch_compile,
            enable_numexpr,
            enable_openblas
        )
        
        # Initialize performance metrics collector
        self.perf_collector = perf_metrics.PerfMetricsCollector()
    
    def _setup_optimizations(
        self,
        enable_torch_compile: bool,
        enable_numexpr: bool,
        enable_openblas: bool
    ):
        """Enhanced optimization setup."""
        if enable_torch_compile:
            # Advanced TorchInductor settings
            torch._dynamo.config.dynamic_shapes = True
            torch._dynamo.config.automatic_dynamic_shapes = True
            torch._inductor.config.cpp.enable_kernel_profile = True
            torch._inductor.config.cpp.enable_kernel_profile = True
            torch._inductor.config.triton.autotune = True
            
            # Enable SIMD
            if self.cpu_optimizer.simd_features["AVX512"]:
                torch._C._jit_set_profiling_executor(True)
            
        if enable_numexpr:
            # Optimize NumExpr
            ne.set_num_threads(self.config.thread_pool_size)
            ne.set_vml_num_threads(self.config.thread_pool_size)
            
        if enable_openblas:
            # Optimize OpenBLAS
            os.environ['OPENBLAS_NUM_THREADS'] = str(self.config.thread_pool_size)
            os.environ['MKL_NUM_THREADS'] = str(self.config.thread_pool_size)
    
    def get_current_stats(self) -> CPUStats:
        """Get enhanced CPU and memory statistics."""
        basic_stats = super().get_current_stats()
        
        # Get cache statistics
        cache_stats = self.perf_collector.get_cache_stats()
        
        # Get instruction mix
        instruction_mix = self.perf_collector.get_instruction_mix()
        
        # Get memory bandwidth
        memory_bandwidth = self.perf_collector.get_memory_bandwidth()
        
        # Get additional metrics
        process = psutil.Process()
        page_faults = process.memory_full_info().num_page_faults
        context_switches = process.num_ctx_switches()
        
        return CPUStats(
            **basic_stats.__dict__,
            cache_stats=cache_stats,
            instruction_mix=instruction_mix,
            memory_bandwidth=memory_bandwidth,
            page_faults=page_faults,
            context_switches=context_switches.voluntary + context_switches.involuntary
        )
    
    def adjust_batch_size(self, sample_size_bytes: int) -> int:
        """Dynamically adjust batch size based on current system load."""
        current_time = time.time()
        if current_time - self.last_adjustment_time < self.config.cooldown_seconds:
            return self.current_batch_size
        
        stats = self.get_current_stats()
        
        # Record metrics
        self.metrics_data.append({
            'timestamp': current_time,
            'batch_size': self.current_batch_size,
            'cpu_percent': stats.cpu_percent,
            'memory_percent': stats.memory_percent,
            'available_memory_gb': stats.available_memory_gb,
            'load_avg_1min': stats.load_average[0],
            'temperature': stats.temperature,
            'power_consumption': stats.power_consumption
        })
        
        # Calculate adjustment factors
        cpu_factor = (self.config.target_cpu_util - stats.cpu_percent) / 100
        memory_factor = (self.config.target_memory_util - stats.memory_percent) / 100
        
        # Combine factors with weights
        adjustment = np.clip(
            min(cpu_factor, memory_factor) * self.config.adjustment_rate,
            -0.5,  # Max 50% decrease
            0.5    # Max 50% increase
        )
        
        # Calculate new batch size
        new_batch_size = int(
            self.current_batch_size * (1 + adjustment)
        )
        
        # Ensure within bounds and memory constraints
        max_by_memory = int(
            stats.available_memory_gb * 0.7 * 1024**3 / sample_size_bytes
        )
        new_batch_size = np.clip(
            new_batch_size,
            self.config.min_batch_size,
            min(self.config.max_batch_size, max_by_memory)
        )
        
        self.current_batch_size = new_batch_size
        self.last_adjustment_time = current_time
        
        return new_batch_size
    
    @contextmanager
    def optimize_batch_processing(self, model: torch.nn.Module, batch_size: int):
        """Context manager for optimized batch processing."""
        try:
            # Enable SIMD optimizations
            optimized_model = self.cpu_optimizer.enable_simd(model)
            
            # Start performance monitoring
            with self.perf_collector.monitor():
                yield optimized_model
        finally:
            # Cleanup
            self.perf_collector.reset()
    
    def export_metrics(self):
        """Export enhanced metrics."""
        if not self.metrics_data:
            logger.warning("No metrics data to export")
            return
        
        # Create DataFrame with basic metrics
        df = pd.DataFrame(self.metrics_data)
        
        # Add advanced metrics
        advanced_metrics = {
            'cache_hit_rates': self.perf_collector.get_cache_stats(),
            'instruction_profile': self.perf_collector.get_instruction_mix(),
            'memory_bandwidth': self.perf_collector.get_memory_bandwidth(),
            'simd_utilization': self.perf_collector.get_simd_utilization()
        }
        
        # Export to CSV
        df.to_csv(self.metrics_file, index=False)
        
        # Export advanced metrics to JSON
        advanced_metrics_file = self.metrics_file.with_suffix('.advanced.json')
        with open(advanced_metrics_file, 'w') as f:
            json.dump(advanced_metrics, f, indent=2)
        
        logger.info(f"Exported metrics to {self.metrics_file}")
        logger.info(f"Exported advanced metrics to {advanced_metrics_file}")
        
        return {**df.mean().to_dict(), **advanced_metrics}

def create_default_scheduler(
    metrics_file: str = "cpu_metrics.csv",
    enable_optimizations: bool = True
) -> CPUScheduler:
    """Create an enhanced CPUScheduler with default configuration."""
    config = BatchSizeConfig(
        min_batch_size=1,
        max_batch_size=512,
        target_cpu_util=80.0,
        target_memory_util=85.0,
        adjustment_rate=0.1,
        cooldown_seconds=5.0,
        prefetch_factor=2,
        cache_warmup_iterations=3,
        simd_threshold=0.8
    )
    
    return CPUScheduler(
        config=config,
        metrics_file=metrics_file,
        enable_torch_compile=enable_optimizations,
        enable_numexpr=enable_optimizations,
        enable_openblas=enable_optimizations
    ) 