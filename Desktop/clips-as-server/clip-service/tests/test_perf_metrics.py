import pytest
import numpy as np
import time
from perf_metrics import PerfMetricsCollector

def test_perf_metrics_initialization():
    collector = PerfMetricsCollector()
    assert collector.cache_hits == 0
    assert collector.cache_misses == 0
    assert collector.instruction_count == 0

def test_metrics_collection():
    collector = PerfMetricsCollector()
    
    # Simulate some work
    with collector:
        # CPU-intensive work
        for _ in range(1000000):
            _ = np.random.random()
        
        # Memory-intensive work
        large_array = np.zeros((1000, 1000))
        for _ in range(100):
            large_array += np.random.random((1000, 1000))
            
        # Cache simulation
        collector.cache_hits += 50
        collector.cache_misses += 10
        
        time.sleep(1)  # Ensure we have some measurable duration
    
    # Verify metrics
    cache_stats = collector.get_cache_stats()
    assert cache_stats['hit_rate'] == pytest.approx(0.833, rel=1e-3)
    assert cache_stats['miss_rate'] == pytest.approx(0.167, rel=1e-3)
    
    # Memory bandwidth should be non-zero
    assert collector.get_avg_memory_bandwidth() > 0
    
    # SIMD utilization should be between 0 and 1
    simd_util = collector.get_simd_utilization()
    assert 0 <= simd_util <= 1
    
    # Instruction profile should have valid counts
    instr_profile = collector.get_instruction_profile()
    assert isinstance(instr_profile, dict)
    assert all(isinstance(v, (int, float)) for v in instr_profile.values())

def test_metrics_reset():
    collector = PerfMetricsCollector()
    
    with collector:
        collector.cache_hits += 10
        collector.cache_misses += 5
        time.sleep(0.1)
    
    collector._reset_metrics()
    assert collector.cache_hits == 0
    assert collector.cache_misses == 0
    assert collector.instruction_count == 0

def test_multiple_collection_periods():
    collector = PerfMetricsCollector()
    
    # First collection period
    with collector:
        collector.cache_hits += 20
        time.sleep(0.1)
    
    first_stats = collector.get_cache_stats()
    
    # Second collection period
    with collector:
        collector.cache_hits += 30
        collector.cache_misses += 10
        time.sleep(0.1)
    
    second_stats = collector.get_cache_stats()
    
    assert second_stats['hit_rate'] > first_stats['hit_rate']
    assert second_stats['total_ops'] > first_stats['total_ops']

def test_error_handling():
    collector = PerfMetricsCollector()
    
    with pytest.raises(RuntimeError):
        # Try to get metrics without starting collection
        collector.get_avg_memory_bandwidth()
    
    with pytest.raises(RuntimeError):
        # Try to get metrics without starting collection
        collector.get_simd_utilization() 