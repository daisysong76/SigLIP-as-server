import os
import time
import psutil
import numpy as np
from contextlib import contextmanager
from typing import Dict, Union, Optional

class PerfMetricsCollector:
    """Performance metrics collector for monitoring cache, memory, and CPU utilization."""
    
    def __init__(self):
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.instruction_count: int = 0
        
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._process = psutil.Process()
        self._initial_cpu_times = None
        self._initial_memory = None
        self._collection_active = False
        
    def __enter__(self):
        """Start collecting performance metrics."""
        self._start_collection()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop collecting performance metrics."""
        self._end_collection()
        
    def _start_collection(self):
        """Initialize collection of performance metrics."""
        self._start_time = time.time()
        self._initial_cpu_times = self._process.cpu_times()
        self._initial_memory = self._process.memory_info()
        self._collection_active = True
        
    def _end_collection(self):
        """Finalize collection of performance metrics."""
        self._end_time = time.time()
        self._collection_active = False
        
    def _reset_metrics(self):
        """Reset all collected metrics."""
        self.cache_hits = 0
        self.cache_misses = 0
        self.instruction_count = 0
        self._start_time = None
        self._end_time = None
        self._initial_cpu_times = None
        self._initial_memory = None
        self._collection_active = False
        
    def _check_collection_active(self):
        """Verify that metrics collection is active."""
        if not self._collection_active:
            raise RuntimeError("Metrics collection is not active. Use 'with' statement to collect metrics.")
            
    def get_cache_stats(self) -> Dict[str, float]:
        """Get cache statistics including hit rate and miss rate."""
        self._check_collection_active()
        total_ops = self.cache_hits + self.cache_misses
        if total_ops == 0:
            return {
                'hit_rate': 0.0,
                'miss_rate': 0.0,
                'total_ops': 0
            }
            
        return {
            'hit_rate': self.cache_hits / total_ops,
            'miss_rate': self.cache_misses / total_ops,
            'total_ops': total_ops
        }
        
    def get_avg_memory_bandwidth(self) -> float:
        """Calculate average memory bandwidth in MB/s."""
        self._check_collection_active()
        current_memory = self._process.memory_info()
        memory_delta = (current_memory.rss - self._initial_memory.rss) / 1024 / 1024  # Convert to MB
        time_delta = time.time() - self._start_time
        return memory_delta / time_delta if time_delta > 0 else 0
        
    def get_simd_utilization(self) -> float:
        """Estimate SIMD utilization based on CPU metrics."""
        self._check_collection_active()
        current_cpu_times = self._process.cpu_times()
        user_time_delta = current_cpu_times.user - self._initial_cpu_times.user
        system_time_delta = current_cpu_times.system - self._initial_cpu_times.system
        total_time = user_time_delta + system_time_delta
        
        if total_time == 0:
            return 0.0
            
        # Estimate SIMD utilization based on user time ratio
        # This is a simplified approximation
        return user_time_delta / total_time if total_time > 0 else 0
        
    def get_instruction_profile(self) -> Dict[str, Union[int, float]]:
        """Get instruction execution profile."""
        self._check_collection_active()
        current_cpu_times = self._process.cpu_times()
        
        return {
            'user_time': current_cpu_times.user - self._initial_cpu_times.user,
            'system_time': current_cpu_times.system - self._initial_cpu_times.system,
            'instruction_count': self.instruction_count,
            'instructions_per_second': (
                self.instruction_count / (time.time() - self._start_time)
                if time.time() > self._start_time
                else 0
            )
        } 