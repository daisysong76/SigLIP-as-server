#!/usr/bin/env python3
"""
Dynamic batch sizing utility for optimal inference throughput.
"""

import time
import torch
import logging
import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BatchPerformanceMetrics:
    """Metrics for a single batch processing run."""
    timestamp: float
    batch_size: int
    processing_time: float  # in seconds
    items_per_second: float
    memory_usage: Optional[float] = None  # in percentage (0-1)
    

class DynamicBatchSizer:
    """
    Dynamically adjusts batch size based on performance metrics and hardware constraints.
    Works with any model type (CLIP, LLaVA, etc.)
    """
    
    def __init__(
        self,
        initial_batch_size: int = 4,
        min_batch_size: int = 1,
        max_batch_size: int = 32,
        target_memory_usage: float = 0.7,  # Keep memory usage around 70%
        target_latency: Optional[float] = None,  # Target processing time (s) per batch
        memory_headroom: float = 0.1,  # Extra memory buffer to prevent OOM
        window_size: int = 10,  # Number of recent batches to consider
        device: str = "cuda",
        model_name: str = "unknown"
    ):
        """
        Initialize with batch sizing parameters.
        
        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum batch size to use
            max_batch_size: Maximum batch size to use
            target_memory_usage: Target memory usage as percentage (0-1)
            target_latency: Optional target latency per batch
            memory_headroom: Extra memory buffer to prevent OOM errors
            window_size: Number of recent batches to consider for trends
            device: Device to run on ('cuda', 'cpu', etc.)
            model_name: Name of model using this batch sizer
        """
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_memory_usage = target_memory_usage
        self.target_latency = target_latency
        self.memory_headroom = memory_headroom
        self.window_size = window_size
        self.device = device
        self.model_name = model_name
        
        # Current batch size
        self.batch_size = initial_batch_size
        
        # Performance history
        self.history: List[BatchPerformanceMetrics] = []
        self.iteration_count = 0
        
        # Flags for auto-tuning
        self.memory_limited = False
        self.latency_limited = False
        
        # Log device information
        self.is_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        self._log_device_info()
    
    def _log_device_info(self):
        """Log information about the hardware."""
        try:
            if self.is_cuda:
                device_props = torch.cuda.get_device_properties(0)
                logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
                logger.info(f"Total memory: {device_props.total_memory / 1e9:.2f} GB")
                logger.info(f"Compute capability: {device_props.major}.{device_props.minor}")
            else:
                import multiprocessing
                logger.info(f"CPU Device: {multiprocessing.cpu_count()} cores")
        except Exception as e:
            logger.warning(f"Failed to log device info: {e}")
    
    def get_batch_size(self) -> int:
        """Get the current optimal batch size."""
        return self.batch_size
    
    def get_optimal_workers(self) -> int:
        """Get optimal number of workers for data loading."""
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        
        if self.is_cuda:
            # For CUDA, use fewer workers to avoid CPU bottleneck
            return min(4, max(2, cpu_count // 2))
        else:
            # For CPU, use more workers but leave some cores free
            return max(1, cpu_count - 1)
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage as a percentage (0-1)."""
        if self.is_cuda:
            # Get GPU memory usage
            memory_allocated = torch.cuda.memory_allocated()
            memory_reserved = torch.cuda.memory_reserved()
            device_props = torch.cuda.get_device_properties(0)
            total_memory = device_props.total_memory
            
            # Use the maximum of allocated and reserved memory
            return max(memory_allocated, memory_reserved) / total_memory
        else:
            # Get CPU memory usage
            import psutil
            return psutil.virtual_memory().percent / 100.0
    
    def _get_recent_metrics(self, window: int = None) -> List[BatchPerformanceMetrics]:
        """Get the most recent batch metrics."""
        window = window or self.window_size
        return self.history[-window:] if len(self.history) > 0 else []
    
    def record_batch_metrics(
        self, 
        batch_size: int, 
        processing_time: float, 
        memory_usage: Optional[float] = None
    ) -> BatchPerformanceMetrics:
        """
        Record metrics for a processed batch.
        
        Args:
            batch_size: Size of the batch that was processed
            processing_time: Time in seconds to process the batch
            memory_usage: Optional memory usage measurement
            
        Returns:
            The recorded metrics object
        """
        # Calculate items per second
        items_per_second = batch_size / max(processing_time, 1e-6)
        
        # Get memory usage if not provided
        if memory_usage is None and self.is_cuda:
            memory_usage = self._get_current_memory_usage()
        
        # Create metrics object
        metrics = BatchPerformanceMetrics(
            timestamp=time.time(),
            batch_size=batch_size,
            processing_time=processing_time,
            items_per_second=items_per_second,
            memory_usage=memory_usage
        )
        
        # Record metrics
        self.history.append(metrics)
        
        # Keep history bounded
        if len(self.history) > self.window_size * 10:
            self.history = self.history[-self.window_size * 5:]
        
        return metrics
    
    def adjust_batch_size(
        self, 
        processing_time: float, 
        batch_size: Optional[int] = None, 
        memory_usage: Optional[float] = None
    ) -> int:
        """
        Adjust the batch size based on performance metrics.
        
        Args:
            processing_time: Time in seconds to process the last batch
            batch_size: Optional size of the batch that was processed
            memory_usage: Optional memory usage measurement
            
        Returns:
            New batch size to use
        """
        # Use current batch size if not specified
        batch_size = batch_size or self.batch_size
        
        # Record metrics for this batch
        metrics = self.record_batch_metrics(
            batch_size=batch_size,
            processing_time=processing_time,
            memory_usage=memory_usage
        )
        
        # Increment iteration counter
        self.iteration_count += 1
        
        # Start with current batch size
        new_batch_size = self.batch_size
        adjustment_reason = ""
        
        # Check memory constraints first (highest priority)
        if self.is_cuda and metrics.memory_usage is not None:
            # If memory usage is above target, reduce batch size
            if metrics.memory_usage > self.target_memory_usage:
                # Calculate reduction factor based on how far above target
                overage_ratio = metrics.memory_usage / self.target_memory_usage
                # More aggressive reduction for higher memory usage
                reduction_factor = 0.8 / max(1.0, overage_ratio)
                new_batch_size = max(self.min_batch_size, int(new_batch_size * reduction_factor))
                adjustment_reason = f"high memory usage ({metrics.memory_usage:.1%} > {self.target_memory_usage:.1%})"
                self.memory_limited = True
            # If memory usage is well below target and we were previously memory-limited
            elif metrics.memory_usage < (self.target_memory_usage - self.memory_headroom) and self.memory_limited:
                # Room to increase batch size
                headroom = self.target_memory_usage - metrics.memory_usage
                increase_factor = min(1.2, 1.0 + headroom)
                new_batch_size = min(self.max_batch_size, int(new_batch_size * increase_factor))
                adjustment_reason = f"low memory usage ({metrics.memory_usage:.1%} << {self.target_memory_usage:.1%})"
                # Only reset memory_limited if we're well below the target
                if metrics.memory_usage < self.target_memory_usage * 0.7:
                    self.memory_limited = False
        
        # Check latency constraints if specified and no memory adjustment
        if not adjustment_reason and self.target_latency is not None:
            if processing_time > self.target_latency:
                # Reduce batch size to meet latency target
                ratio = processing_time / self.target_latency
                reduction_factor = 0.9 / max(1.0, ratio)
                new_batch_size = max(self.min_batch_size, int(new_batch_size * reduction_factor))
                adjustment_reason = f"high latency ({processing_time:.3f}s > {self.target_latency:.3f}s)"
                self.latency_limited = True
            elif processing_time < self.target_latency * 0.7 and self.latency_limited:
                # Increase batch size if latency is well below target
                headroom = 1.0 - (processing_time / self.target_latency)
                increase_factor = min(1.2, 1.0 + headroom * 0.5)
                new_batch_size = min(self.max_batch_size, int(new_batch_size * increase_factor))
                adjustment_reason = f"low latency ({processing_time:.3f}s << {self.target_latency:.3f}s)"
                # Only reset latency_limited if we're well below the target
                if processing_time < self.target_latency * 0.5:
                    self.latency_limited = False
        
        # If no adjustments made yet, check throughput trends
        if not adjustment_reason and self.iteration_count > 2:
            recent_metrics = self._get_recent_metrics(min(self.window_size, self.iteration_count - 1))
            if recent_metrics:
                # Calculate average items/second from recent batches
                avg_items_per_sec = np.mean([m.items_per_second for m in recent_metrics])
                
                # Get items/second from this batch
                current_items_per_sec = metrics.items_per_second
                
                # If current throughput is better than average, try increasing batch size
                if current_items_per_sec > avg_items_per_sec * 1.1 and new_batch_size < self.max_batch_size:
                    new_batch_size = min(self.max_batch_size, new_batch_size + 1)
                    adjustment_reason = "increasing throughput"
                # If current throughput is worse than average, consider decreasing
                elif current_items_per_sec < avg_items_per_sec * 0.9 and new_batch_size > self.min_batch_size:
                    new_batch_size = max(self.min_batch_size, new_batch_size - 1)
                    adjustment_reason = "decreasing throughput"
        
        # Log adjustment if there was one
        if new_batch_size != self.batch_size:
            logger.info(f"[{self.model_name}] Batch size adjusted from {self.batch_size} to {new_batch_size} ({adjustment_reason})")
            self.batch_size = new_batch_size
        
        return new_batch_size
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        recent_metrics = self._get_recent_metrics()
        
        if not recent_metrics:
            return {
                "batch_size": self.batch_size,
                "memory_usage": None,
                "items_per_second": None,
                "processing_time": None,
                "memory_limited": self.memory_limited,
                "latency_limited": self.latency_limited
            }
        
        # Calculate average metrics
        avg_batch_size = np.mean([m.batch_size for m in recent_metrics])
        avg_processing_time = np.mean([m.processing_time for m in recent_metrics])
        avg_items_per_second = np.mean([m.items_per_second for m in recent_metrics])
        
        # Memory usage (if available)
        memory_values = [m.memory_usage for m in recent_metrics if m.memory_usage is not None]
        avg_memory_usage = np.mean(memory_values) if memory_values else None
        
        return {
            "batch_size": self.batch_size,
            "avg_batch_size": avg_batch_size,
            "memory_usage": avg_memory_usage,
            "items_per_second": avg_items_per_second,
            "processing_time": avg_processing_time,
            "memory_limited": self.memory_limited,
            "latency_limited": self.latency_limited,
            "history_samples": len(self.history)
        }
    
    def save_metrics(self, filepath: Optional[str] = None) -> Optional[str]:
        """
        Save metrics to a JSON file.
        
        Args:
            filepath: Optional path to save metrics to
            
        Returns:
            Path to the metrics file if saved
        """
        if not self.history:
            logger.warning("No metrics to save")
            return None
        
        try:
            import json
            import datetime
            from pathlib import Path
            
            # Create default filename if not provided
            if filepath is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"metrics_{self.model_name}_{timestamp}.json"
            
            # Create directory if necessary
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert metrics to serializable format
            metrics_data = {
                "model_name": self.model_name,
                "device": self.device,
                "current_batch_size": self.batch_size,
                "min_batch_size": self.min_batch_size,
                "max_batch_size": self.max_batch_size,
                "target_memory_usage": self.target_memory_usage,
                "target_latency": self.target_latency,
                "metrics": self.get_metrics(),
                "history": [
                    {
                        "timestamp": m.timestamp,
                        "batch_size": m.batch_size,
                        "processing_time": m.processing_time,
                        "items_per_second": m.items_per_second,
                        "memory_usage": m.memory_usage
                    }
                    for m in self.history
                ]
            }
            
            # Save to file
            with open(filepath, "w") as f:
                json.dump(metrics_data, f, indent=2)
            
            logger.info(f"Metrics saved to {filepath}")
            return str(filepath)
        
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            return None 