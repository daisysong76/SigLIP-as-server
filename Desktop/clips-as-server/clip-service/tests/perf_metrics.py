"""Performance metrics collector for CPU profiling."""

import psutil
import numpy as np
from typing import Dict, Optional
import time
from contextlib import contextmanager
import platform
import os
from dataclasses import dataclass
from collections import deque
import logging
import ctypes
from typing import List, Tuple

logger = logging.getLogger(__name__)

@dataclass
class CacheMetrics:
    """Cache-related performance metrics."""
    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    l3_hits: int = 0
    l3_misses: int = 0

@dataclass
class InstructionMetrics:
    """Instruction-related performance metrics."""
    total_instructions: int = 0
    simd_instructions: int = 0
    memory_instructions: int = 0
    scalar_instructions: int = 0
    branches: int = 0
    branch_misses: int = 0

class PerfMetricsCollector:
    """Collects and analyzes CPU performance metrics."""
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.cache_metrics = deque(maxlen=history_size)
        self.instruction_metrics = deque(maxlen=history_size)
        self.memory_bandwidth_history = deque(maxlen=history_size)
        self.simd_utilization_history = deque(maxlen=history_size)
        
        # Initialize platform-specific performance counters
        self._init_perf_counters()
    
    def _init_perf_counters(self):
        """Initialize platform-specific performance counters."""
        self.has_perf_counters = False
        try:
            if platform.system() == "Linux":
                # Try to load Linux perf event API
                self.libperf = ctypes.CDLL('libperf.so', mode=ctypes.RTLD_GLOBAL)
                self.has_perf_counters = True
            elif platform.system() == "Darwin":
                # macOS performance counters
                self.has_perf_counters = False  # Currently not implemented
            else:
                logger.warning(f"Performance counters not supported on {platform.system()}")
        except Exception as e:
            logger.warning(f"Failed to initialize performance counters: {e}")
    
    def _read_cache_events(self) -> CacheMetrics:
        """Read cache-related performance events."""
        if not self.has_perf_counters:
            return CacheMetrics()
        
        try:
            if platform.system() == "Linux":
                # Read perf events for cache statistics
                metrics = CacheMetrics()
                
                # Example of reading from perf events (simplified)
                with open('/proc/self/stat', 'r') as f:
                    stats = f.read().split()
                    # These indices are examples and may need adjustment
                    metrics.l1_misses = int(stats[9])
                    metrics.l2_misses = int(stats[11])
                
                return metrics
        except Exception as e:
            logger.debug(f"Failed to read cache events: {e}")
        
        return CacheMetrics()
    
    def _read_instruction_events(self) -> InstructionMetrics:
        """Read instruction-related performance events."""
        if not self.has_perf_counters:
            return InstructionMetrics()
        
        try:
            if platform.system() == "Linux":
                metrics = InstructionMetrics()
                
                # Example of reading instruction-related events
                with open('/proc/self/stat', 'r') as f:
                    stats = f.read().split()
                    metrics.total_instructions = int(stats[10])
                    metrics.branches = int(stats[12])
                    metrics.branch_misses = int(stats[13])
                
                return metrics
        except Exception as e:
            logger.debug(f"Failed to read instruction events: {e}")
        
        return InstructionMetrics()
    
    def _calculate_memory_bandwidth(self) -> float:
        """Calculate memory bandwidth utilization."""
        try:
            process = psutil.Process()
            # Get memory info before and after a small delay
            mem_info1 = process.memory_full_info()
            time.sleep(0.1)
            mem_info2 = process.memory_full_info()
            
            # Calculate bandwidth (bytes/second)
            bytes_delta = abs(mem_info2.rss - mem_info1.rss)
            bandwidth = bytes_delta / 0.1  # Convert to bytes per second
            
            return bandwidth / (1024 * 1024)  # Convert to MB/s
        except Exception as e:
            logger.debug(f"Failed to calculate memory bandwidth: {e}")
            return 0.0
    
    def _estimate_simd_utilization(self) -> float:
        """Estimate SIMD instruction utilization."""
        if not self.has_perf_counters:
            return 0.0
        
        try:
            metrics = self._read_instruction_events()
            if metrics.total_instructions > 0:
                return metrics.simd_instructions / metrics.total_instructions
        except Exception as e:
            logger.debug(f"Failed to estimate SIMD utilization: {e}")
        
        return 0.0
    
    @contextmanager
    def monitor(self):
        """Context manager for monitoring performance metrics."""
        start_time = time.time()
        start_cache = self._read_cache_events()
        start_instructions = self._read_instruction_events()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_cache = self._read_cache_events()
            end_instructions = self._read_instruction_events()
            
            # Calculate deltas
            duration = end_time - start_time
            
            # Cache metrics
            cache_delta = CacheMetrics(
                l1_hits=end_cache.l1_hits - start_cache.l1_hits,
                l1_misses=end_cache.l1_misses - start_cache.l1_misses,
                l2_hits=end_cache.l2_hits - start_cache.l2_hits,
                l2_misses=end_cache.l2_misses - start_cache.l2_misses,
                l3_hits=end_cache.l3_hits - start_cache.l3_hits,
                l3_misses=end_cache.l3_misses - start_cache.l3_misses
            )
            self.cache_metrics.append(cache_delta)
            
            # Instruction metrics
            instruction_delta = InstructionMetrics(
                total_instructions=end_instructions.total_instructions - start_instructions.total_instructions,
                simd_instructions=end_instructions.simd_instructions - start_instructions.simd_instructions,
                memory_instructions=end_instructions.memory_instructions - start_instructions.memory_instructions,
                scalar_instructions=end_instructions.scalar_instructions - start_instructions.scalar_instructions,
                branches=end_instructions.branches - start_instructions.branches,
                branch_misses=end_instructions.branch_misses - start_instructions.branch_misses
            )
            self.instruction_metrics.append(instruction_delta)
            
            # Memory bandwidth
            bandwidth = self._calculate_memory_bandwidth()
            self.memory_bandwidth_history.append(bandwidth)
            
            # SIMD utilization
            simd_util = self._estimate_simd_utilization()
            self.simd_utilization_history.append(simd_util)
    
    def get_cache_stats(self) -> Dict[str, float]:
        """Get cache statistics."""
        if not self.cache_metrics:
            return {'l1_hit_rate': 0.0, 'l2_hit_rate': 0.0, 'l3_hit_rate': 0.0}
        
        # Calculate average hit rates
        metrics = self.cache_metrics[-1]  # Get most recent metrics
        
        l1_total = metrics.l1_hits + metrics.l1_misses
        l2_total = metrics.l2_hits + metrics.l2_misses
        l3_total = metrics.l3_hits + metrics.l3_misses
        
        return {
            'l1_hit_rate': metrics.l1_hits / l1_total if l1_total > 0 else 0.0,
            'l2_hit_rate': metrics.l2_hits / l2_total if l2_total > 0 else 0.0,
            'l3_hit_rate': metrics.l3_hits / l3_total if l3_total > 0 else 0.0
        }
    
    def get_instruction_mix(self) -> Dict[str, float]:
        """Get instruction mix profile."""
        if not self.instruction_metrics:
            return {
                'simd_ratio': 0.0,
                'memory_ratio': 0.0,
                'scalar_ratio': 0.0,
                'branch_miss_rate': 0.0
            }
        
        metrics = self.instruction_metrics[-1]  # Get most recent metrics
        total = metrics.total_instructions
        
        if total == 0:
            return {
                'simd_ratio': 0.0,
                'memory_ratio': 0.0,
                'scalar_ratio': 0.0,
                'branch_miss_rate': 0.0
            }
        
        return {
            'simd_ratio': metrics.simd_instructions / total,
            'memory_ratio': metrics.memory_instructions / total,
            'scalar_ratio': metrics.scalar_instructions / total,
            'branch_miss_rate': metrics.branch_misses / metrics.branches if metrics.branches > 0 else 0.0
        }
    
    def get_memory_bandwidth(self) -> float:
        """Get average memory bandwidth (MB/s)."""
        if not self.memory_bandwidth_history:
            return 0.0
        return np.mean(list(self.memory_bandwidth_history))
    
    def get_simd_utilization(self) -> float:
        """Get SIMD utilization ratio."""
        if not self.simd_utilization_history:
            return 0.0
        return np.mean(list(self.simd_utilization_history))
    
    def reset(self):
        """Reset all metrics."""
        self.cache_metrics.clear()
        self.instruction_metrics.clear()
        self.memory_bandwidth_history.clear()
        self.simd_utilization_history.clear() 