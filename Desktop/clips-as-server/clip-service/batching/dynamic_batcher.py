#!/usr/bin/env python3
"""
Dynamic batching system that adapts batch sizes based on hardware capabilities and performance metrics.
"""

import os
import time
import json
import logging
import functools
import numpy as np
import psutil
import gc
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from functools import wraps
import threading
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DynamicBatcher")

@dataclass
class BatchMetrics:
    """Class to store batch processing metrics"""
    timestamp: float
    batch_size: int
    processing_time: float
    memory_usage: float  # in GB
    batch_shape: Tuple
    throughput: float


class DynamicBatcher:
    """
    Dynamic batch size controller that adjusts batch size based on processing time and memory usage.
    
    Features:
    - Adapts batch size based on hardware capabilities and input characteristics
    - Monitors and adapts to memory usage to prevent OOM errors
    - Optimizes for throughput while respecting latency constraints
    - Can automatically tune growth/reduction factors based on performance
    - Tracks metrics for analysis and visualization
    """
    
    def __init__(
        self,
        initial_batch_size: int = 16,
        min_batch_size: int = 1,
        max_batch_size: int = 256,
        target_memory_usage: float = 0.7,  # Target memory usage (0-1)
        target_latency: Optional[float] = None,  # Target latency in seconds
        growth_factor: float = 1.2,
        reduction_factor: float = 0.7,
        memory_headroom: float = 0.1,  # Free memory buffer
        window_size: int = 10,  # Number of batches to consider for auto-tuning
        auto_tune: bool = True,  # Whether to auto-tune growth/reduction factors
        auto_tune_interval: int = 50,  # How often to auto-tune (in batches)
        enable_monitoring: bool = True,  # Whether to record detailed metrics
        device: str = 'cuda'
    ):
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_memory_usage = target_memory_usage
        self.target_latency = target_latency
        self.growth_factor = growth_factor
        self.reduction_factor = reduction_factor
        self.memory_headroom = memory_headroom
        self.window_size = window_size
        self.auto_tune = auto_tune
        self.auto_tune_interval = auto_tune_interval
        self.enable_monitoring = enable_monitoring
        self.device = device
        
        # Current batch size
        self.batch_size = initial_batch_size
        
        # Metrics tracking
        self.history: List[BatchMetrics] = []
        self.iteration_count = 0
        
        # Auto-tuning metrics
        self.last_throughput = 0.0
        self.best_throughput = 0.0
        self.best_batch_size = initial_batch_size
        self.consecutive_decreases = 0
        self.consecutive_increases = 0
        
        # Device info
        self.is_cuda = device == 'cuda' and torch.cuda.is_available()
        
        # Log capabilities
        self._log_capabilities()
    
    def _log_capabilities(self):
        """Log hardware capabilities"""
        logger.info(f"Dynamic batcher initialized with:")
        logger.info(f"  Initial batch size: {self.initial_batch_size}")
        logger.info(f"  Min/Max batch size: {self.min_batch_size}/{self.max_batch_size}")
        logger.info(f"  Target memory usage: {self.target_memory_usage:.1%}")
        if self.target_latency:
            logger.info(f"  Target latency: {self.target_latency:.4f}s")
        
        # Log CPU info
        cpu_count = psutil.cpu_count(logical=False)
        cpu_logical_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            cpu_freq_str = f"{cpu_freq.current:.2f} MHz"
        else:
            cpu_freq_str = "N/A"
        
        logger.info(f"CPU: {cpu_count} physical cores, {cpu_logical_count} logical cores, {cpu_freq_str}")
        
        # Log memory info
        memory = psutil.virtual_memory()
        logger.info(f"RAM: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
        
        # Log GPU info if available
        if self.is_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            gpu_mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            
            logger.info(f"GPU: {gpu_name}")
            logger.info(f"GPU Memory: {gpu_mem_total:.1f} GB total, "
                        f"{gpu_mem_reserved:.1f} GB reserved, "
                        f"{gpu_mem_allocated:.1f} GB allocated")
    
    def get_batch_size(self) -> int:
        """Get current batch size"""
        return self.batch_size
    
    def get_optimal_workers(self) -> int:
        """Get optimal number of workers for DataLoader based on system"""
        # For CUDA, 2-4 workers is usually optimal
        if self.is_cuda:
            return min(4, max(2, psutil.cpu_count(logical=False) // 2))
        # For CPU, use physical cores minus 1 to leave resources for other processes
        else:
            return max(1, psutil.cpu_count(logical=False) - 1)
    
    def _record_metrics(self, processing_time: float, batch_shape: tuple, memory_usage: float) -> None:
        """Record batch processing metrics"""
        if not self.enable_monitoring:
            return
            
        # Calculate throughput (items/second)
        batch_size = self.batch_size
        throughput = batch_size / processing_time if processing_time > 0 else 0
        
        # Record metrics
        metrics = BatchMetrics(
            timestamp=time.time(),
            batch_size=batch_size,
            processing_time=processing_time,
            memory_usage=memory_usage,
            batch_shape=batch_shape,
            throughput=throughput
        )
        
        self.history.append(metrics)
        
        # Keep history within window size for efficiency
        if len(self.history) > self.window_size * 10:
            self.history = self.history[-self.window_size * 10:]
    
    def save_metrics(self, output_file: str) -> str:
        """Save metrics to file"""
        if not self.enable_monitoring or not self.history:
            logger.warning("No metrics to save")
            return ""
        
        # Create metrics dictionary
        metrics_dict = {
            "config": {
                "initial_batch_size": self.initial_batch_size,
                "min_batch_size": self.min_batch_size,
                "max_batch_size": self.max_batch_size,
                "target_memory_usage": self.target_memory_usage,
                "target_latency": self.target_latency,
                "growth_factor": self.growth_factor,
                "reduction_factor": self.reduction_factor,
                "auto_tune": self.auto_tune
            },
            "metrics": []
        }
        
        # Add metrics
        for m in self.history:
            metrics_dict["metrics"].append({
                "timestamp": m.timestamp,
                "batch_size": m.batch_size,
                "processing_time": m.processing_time,
                "memory_usage": m.memory_usage,
                "throughput": m.throughput,
                "shape": list(m.batch_shape) if m.batch_shape else None
            })
        
        # Save to file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        logger.info(f"Saved metrics to {output_file}")
        return output_file
    
    def _get_recent_metrics(self) -> List[BatchMetrics]:
        """Get recent metrics within window"""
        if len(self.history) <= self.window_size:
            return self.history
        return self.history[-self.window_size:]
    
    def _auto_tune_parameters(self) -> None:
        """Auto-tune growth and reduction factors based on performance"""
        if not self.auto_tune or self.iteration_count % self.auto_tune_interval != 0:
            return
        
        recent_metrics = self._get_recent_metrics()
        if not recent_metrics:
            return
        
        # Calculate current throughput
        total_items = sum(m.batch_size for m in recent_metrics)
        total_time = sum(m.processing_time for m in recent_metrics)
        current_throughput = total_items / total_time if total_time > 0 else 0
        
        # Update throughput tracking
        throughput_change = (current_throughput - self.last_throughput) / self.last_throughput if self.last_throughput > 0 else 0
        self.last_throughput = current_throughput
        
        # Track best throughput
        if current_throughput > self.best_throughput:
            self.best_throughput = current_throughput
            self.best_batch_size = self.batch_size
            logger.info(f"New best throughput: {current_throughput:.2f} items/s at batch size {self.batch_size}")
        
        # Adjust factors based on throughput change
        if throughput_change > 0.05:  # Significant improvement
            self.consecutive_increases += 1
            self.consecutive_decreases = 0
            if self.consecutive_increases >= 3:
                # Increase growth factor and reduce reduction factor to be more aggressive
                self.growth_factor = min(1.5, self.growth_factor * 1.05)
                self.reduction_factor = max(0.5, self.reduction_factor * 0.95)
                logger.info(f"Auto-tuned factors: growth={self.growth_factor:.2f}, reduction={self.reduction_factor:.2f}")
                self.consecutive_increases = 0
        elif throughput_change < -0.05:  # Significant decrease
            self.consecutive_decreases += 1
            self.consecutive_increases = 0
            if self.consecutive_decreases >= 3:
                # Decrease growth factor and increase reduction factor to be more conservative
                self.growth_factor = max(1.05, self.growth_factor * 0.95)
                self.reduction_factor = min(0.9, self.reduction_factor * 1.05)
                logger.info(f"Auto-tuned factors: growth={self.growth_factor:.2f}, reduction={self.reduction_factor:.2f}")
                self.consecutive_decreases = 0
    
    def adjust_batch_size(
        self, 
        processing_time: float, 
        batch_shape: Optional[tuple] = None,
        memory_usage: Optional[float] = None
    ) -> int:
        """
        Adjust batch size based on processing time and memory usage
        
        Args:
            processing_time: Time taken to process batch in seconds
            batch_shape: Shape of the processed batch (for logging)
            memory_usage: Memory used during processing in GB (if known)
        
        Returns:
            New batch size
        """
        self.iteration_count += 1
        
        # Record metrics
        if self.enable_monitoring:
            self._record_metrics(processing_time, batch_shape, memory_usage or 0.0)
        
        # Auto-tune parameters if enabled
        self._auto_tune_parameters()
        
        # Start with current batch size
        new_batch_size = self.batch_size
        adjustment_reason = ""
        
        # Check memory usage (prioritize this to avoid OOM)
        if self.is_cuda and memory_usage is not None:
            # Estimate total GPU memory
            total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # Calculate available memory
            memory_usage_ratio = memory_usage / total_gpu_memory
            
            # If memory usage is above target, reduce batch size
            if memory_usage_ratio > self.target_memory_usage:
                # Calculate reduction based on how far above target we are
                ratio = memory_usage_ratio / self.target_memory_usage
                temp_reduction = self.reduction_factor * ratio
                new_batch_size = max(self.min_batch_size, 
                                    int(new_batch_size * temp_reduction))
                adjustment_reason = f"memory usage ({memory_usage_ratio:.1%} > {self.target_memory_usage:.1%})"
            # If memory usage is well below target, we can try increasing
            elif memory_usage_ratio < self.target_memory_usage - self.memory_headroom:
                # Only increase if we're not already at max
                if new_batch_size < self.max_batch_size:
                    # Calculate increase based on how far below target we are
                    headroom = self.target_memory_usage - memory_usage_ratio
                    # More aggressive increase when we have lots of headroom
                    increase_factor = min(self.growth_factor * (1 + headroom), 2.0)
                    
                    # Don't increase if we're constrained by latency
                    if not self.target_latency or processing_time < self.target_latency:
                        new_batch_size = min(self.max_batch_size, 
                                            int(new_batch_size * increase_factor))
                        adjustment_reason = f"low memory usage ({memory_usage_ratio:.1%} << {self.target_memory_usage:.1%})"
        
        # Check latency constraints if we didn't already decrease due to memory
        if not adjustment_reason and self.target_latency is not None:
            if processing_time > self.target_latency:
                # Reduce batch size to meet latency target
                ratio = processing_time / self.target_latency
                new_batch_size = max(self.min_batch_size, 
                                    int(new_batch_size * self.reduction_factor / ratio))
                adjustment_reason = f"high latency ({processing_time:.3f}s > {self.target_latency:.3f}s)"
            elif processing_time < self.target_latency * 0.7:
                # Increase batch size if we're well below latency target
                new_batch_size = min(self.max_batch_size, 
                                    int(new_batch_size * self.growth_factor))
                adjustment_reason = f"low latency ({processing_time:.3f}s << {self.target_latency:.3f}s)"
        
        # If we haven't adjusted based on memory or latency, consider throughput
        if not adjustment_reason:
            recent_metrics = self._get_recent_metrics()
            
            if len(recent_metrics) >= 2:
                # Get throughput trend from recent batches
                recent_throughputs = [m.throughput for m in recent_metrics]
                avg_throughput = sum(recent_throughputs) / len(recent_throughputs)
                
                # If throughput is increasing, keep increasing batch size
                if self.iteration_count > 1 and avg_throughput > self.last_throughput:
                    if new_batch_size < self.max_batch_size:
                        new_batch_size = min(self.max_batch_size, 
                                            int(new_batch_size * self.growth_factor))
                        adjustment_reason = "increasing throughput"
                # If throughput decreased, back off a bit
                elif self.iteration_count > 1 and avg_throughput < self.last_throughput * 0.95:
                    new_batch_size = max(self.min_batch_size, 
                                        int(new_batch_size * self.reduction_factor))
                    adjustment_reason = "decreasing throughput"
                # If throughput is steady, try increasing slightly if not constrained
                elif new_batch_size < self.max_batch_size:
                    # Small increase to explore
                    new_batch_size = min(self.max_batch_size, 
                                        int(new_batch_size * 1.05))
                    adjustment_reason = "exploring higher batch sizes"
        
        # If we're on the first few iterations, be more aggressive about exploring
        if self.iteration_count <= 5 and not adjustment_reason:
            if self.iteration_count == 1:
                # First iteration - start exploring from initial size
                new_batch_size = self.initial_batch_size
                adjustment_reason = "initial exploration"
            elif self.iteration_count <= 3:
                # Early iterations - increase more aggressively
                new_batch_size = min(self.max_batch_size, 
                                    int(new_batch_size * self.growth_factor * 1.2))
                adjustment_reason = "early exploration"
        
        # Ensure we're within bounds
        new_batch_size = max(self.min_batch_size, min(self.max_batch_size, new_batch_size))
        
        # Only log if batch size changed
        if new_batch_size != self.batch_size:
            logger.info(f"Adjusted batch size: {self.batch_size} → {new_batch_size} "
                        f"({adjustment_reason or 'unknown reason'})")
            self.batch_size = new_batch_size
        
        return self.batch_size


def batch_processor(func):
    """
    Decorator to handle dynamic batch processing.
    Usage:
    
    @batch_processor
    def process_batch(batch, batch_size, **kwargs):
        # Process the batch
        return result
    """
    @wraps(func)
    def wrapper(batch, batcher=None, **kwargs):
        batch_size = len(batch) if hasattr(batch, '__len__') else 1
        
        # Record start time
        start_time = time.time()
        
        # Record initial memory
        if batcher and batcher.is_cuda:
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated() / (1024**3)  # GB
        
        # Process batch
        result = func(batch, **kwargs)
        
        # Record end time
        if batcher and batcher.is_cuda:
            torch.cuda.synchronize()
        processing_time = time.time() - start_time
        
        # Record memory used
        if batcher and batcher.is_cuda:
            memory_after = torch.cuda.memory_allocated() / (1024**3)  # GB
            memory_usage = memory_after - memory_before
        else:
            memory_usage = None
        
        # Get batch shape if available
        if hasattr(batch, 'shape'):
            batch_shape = batch.shape
        elif isinstance(batch, (list, tuple)) and batch and hasattr(batch[0], 'shape'):
            batch_shape = (len(batch), *batch[0].shape)
        else:
            batch_shape = None
        
        # Adjust batch size if batcher provided
        if batcher:
            batcher.adjust_batch_size(
                processing_time=processing_time,
                batch_shape=batch_shape,
                memory_usage=memory_usage
            )
        
        return result
    
    return wrapper

# Example usage
if __name__ == "__main__":
    # Simple example to demonstrate usage
    batcher = DynamicBatcher(
        initial_batch_size=16,
        min_batch_size=1,
        max_batch_size=64,
        enable_monitoring=True
    )
    
    # Simulate batch processing
    for i in range(20):
        batch_size = batcher.get_batch_size()
        print(f"Processing batch of size {batch_size}")
        
        # Simulate processing time that scales with batch size
        # but has some random variation
        processing_time = 0.01 * batch_size + 0.005 * np.random.random() * batch_size
        memory_usage = 0.2 + (0.5 * batch_size / 64) + 0.1 * np.random.random()
        
        # Adjust for next iteration
        batcher.adjust_batch_size(processing_time, memory_usage=memory_usage)
        
        # Print metrics every 5 batches
        if (i + 1) % 5 == 0:
            metrics = batcher.get_metrics()
            print(f"Recent throughput: {metrics['recent_throughput']:.2f} samples/s")
    
    # Save metrics at the end
    batcher.save_metrics()