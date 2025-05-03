"""
Integration module for DynamicBatchSizer and BatchPerformanceMonitor.
This module provides functionality to seamlessly monitor the DynamicBatchSizer
and generate insights about batch sizing performance.
"""

import logging
import time
import psutil
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Any
import gc
import os
import sys

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import monitoring module
from monitoring.batch_monitor import BatchPerformanceMonitor, monitor_batch

# Import dynamic batcher (assuming it's in the correct location)
try:
    from debug_dataset import DynamicBatchSizer
except ImportError:
    # Try a different location if first import fails
    try:
        from clip_service.debug_dataset import DynamicBatchSizer
    except ImportError:
        raise ImportError("Could not import DynamicBatchSizer. Make sure it's in the correct location.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MonitoredDynamicBatchSizer(DynamicBatchSizer):
    """
    Extension of DynamicBatchSizer that integrates with BatchPerformanceMonitor.
    Provides real-time monitoring, performance visualization, and optimal batch size suggestions.
    """
    
    def __init__(
        self,
        initial_batch_size: int = 16,
        min_batch_size: int = 1,
        max_batch_size: int = 128,
        target_memory_usage: float = 0.85,
        growth_factor: float = 1.5,
        reduction_factor: float = 0.8,
        enable_monitoring: bool = True,
        monitoring_window: int = 100,
        enable_live_plot: bool = True,
        metrics_dir: str = "metrics",
        auto_adjust: bool = True
    ):
        """
        Initialize the MonitoredDynamicBatchSizer.
        
        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            target_memory_usage: Target memory usage ratio (0-1)
            growth_factor: Factor by which to grow batch size
            reduction_factor: Factor by which to reduce batch size
            enable_monitoring: Whether to enable performance monitoring
            monitoring_window: Number of batches to keep in monitoring window
            enable_live_plot: Whether to enable live plotting
            metrics_dir: Directory to save metrics and reports
            auto_adjust: Whether to automatically adjust batch size
        """
        # Initialize parent class
        super().__init__(
            initial_batch_size=initial_batch_size,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            target_memory_usage=target_memory_usage,
            growth_factor=growth_factor,
            reduction_factor=reduction_factor
        )
        
        # Monitoring settings
        self.enable_monitoring = enable_monitoring
        self.auto_adjust = auto_adjust
        
        # Initialize monitor if enabled
        if self.enable_monitoring:
            self.monitor = BatchPerformanceMonitor(
                window_size=monitoring_window,
                enable_live_plot=enable_live_plot,
                save_metrics=True,
                metrics_dir=metrics_dir
            )
            
            # Start live plotting if enabled
            if enable_live_plot:
                self.monitor.start_live_plot()
                
            logger.info(
                f"Monitoring enabled for DynamicBatchSizer with window={monitoring_window}, "
                f"live_plot={enable_live_plot}, auto_adjust={auto_adjust}"
            )
        else:
            self.monitor = None
    
    def adjust_batch_size(self, processing_time: float, memory_usage: Optional[float] = None) -> int:
        """
        Adjust batch size based on processing time and memory usage, with monitoring.
        
        Args:
            processing_time: Time taken to process the last batch (seconds)
            memory_usage: Current memory usage ratio (0-1), measured if None
            
        Returns:
            The new batch size
        """
        # Measure memory usage if not provided
        if memory_usage is None:
            memory_usage = self._get_memory_usage()
        
        # Record batch metrics before adjustment
        if self.enable_monitoring and self.monitor:
            self.monitor.record_batch(
                batch_size=self.batch_size,
                processing_time=processing_time,
                memory_usage=memory_usage
            )
        
        # Adjust batch size if auto-adjust is enabled
        if self.auto_adjust:
            old_batch_size = self.batch_size
            
            # Call parent class method to adjust batch size
            new_batch_size = super().adjust_batch_size(processing_time, memory_usage)
            
            if old_batch_size != new_batch_size:
                logger.info(
                    f"Batch size adjusted: {old_batch_size} → {new_batch_size} "
                    f"(memory: {memory_usage:.2%}, time: {processing_time:.4f}s)"
                )
        else:
            # If auto-adjust is disabled, just return current batch size
            new_batch_size = self.batch_size
            
        return new_batch_size
    
    def get_optimal_batch_size(self) -> int:
        """
        Get the optimal batch size based on performance monitoring data.
        
        Returns:
            Optimal batch size for maximizing throughput
        """
        if not self.enable_monitoring or not self.monitor:
            return self.batch_size
            
        return self.monitor.get_optimal_batch_size()
    
    def apply_optimal_batch_size(self) -> int:
        """
        Apply the optimal batch size from performance data.
        
        Returns:
            The newly applied optimal batch size
        """
        if not self.enable_monitoring or not self.monitor:
            logger.warning("Cannot apply optimal batch size: monitoring not enabled")
            return self.batch_size
            
        optimal = self.monitor.get_optimal_batch_size()
        
        # Ensure optimal size is within bounds
        optimal = max(self.min_batch_size, min(optimal, self.max_batch_size))
        
        if optimal != self.batch_size:
            logger.info(f"Applying optimal batch size: {self.batch_size} → {optimal}")
            self.batch_size = optimal
            
        return optimal
    
    def generate_performance_report(self, save_path: Optional[str] = None) -> Dict:
        """
        Generate a performance report for the batch sizer.
        
        Args:
            save_path: Path to save the report visualization
            
        Returns:
            Dictionary with performance metrics
        """
        if not self.enable_monitoring or not self.monitor:
            logger.warning("Cannot generate report: monitoring not enabled")
            return {"error": "monitoring not enabled"}
            
        # If no save path provided, create one
        if save_path is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_path = f"batch_performance_report_{timestamp}.png"
            
        return self.monitor.generate_report(save_path)
    
    def stop_monitoring(self):
        """Stop monitoring and generate final report."""
        if self.enable_monitoring and self.monitor:
            self.monitor.stop()
            logger.info("Performance monitoring stopped")


# Decorator for process function with monitoring
def monitor_process_function(batch_sizer: MonitoredDynamicBatchSizer):
    """
    Decorator to automatically monitor and adjust batch size for processing functions.
    
    Args:
        batch_sizer: The MonitoredDynamicBatchSizer instance
        
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(batch, *args, **kwargs):
            # Measure processing time
            start_time = time.time()
            
            # Process batch
            result = func(batch, *args, **kwargs)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Measure memory usage
            memory_usage = psutil.virtual_memory().percent / 100.0
            
            # Adjust batch size for next batch
            batch_sizer.adjust_batch_size(processing_time, memory_usage)
            
            return result
        return wrapper
    return decorator


# Example usage function
def run_example(
    num_iterations: int = 100,
    initial_batch_size: int = 8,
    simulated_processing: bool = True,
    apply_optimal_every: int = 20
):
    """
    Run an example of the MonitoredDynamicBatchSizer.
    
    Args:
        num_iterations: Number of batches to process
        initial_batch_size: Initial batch size to use
        simulated_processing: Whether to simulate processing (True) or use real data
        apply_optimal_every: Apply optimal batch size every N iterations
    """
    print(f"Starting MonitoredDynamicBatchSizer example with {num_iterations} iterations...")
    
    # Initialize the monitored batch sizer
    batch_sizer = MonitoredDynamicBatchSizer(
        initial_batch_size=initial_batch_size,
        min_batch_size=1,
        max_batch_size=64,
        enable_monitoring=True,
        enable_live_plot=True,
        auto_adjust=True
    )
    
    # Define a processing function (simulated or real)
    if simulated_processing:
        # Simulated processing function
        def process_batch(batch):
            batch_size = len(batch)
            
            # Simulate processing time scaling with batch size
            # - Small batches: less efficient (overhead dominates)
            # - Medium batches: most efficient (good parallelism, low overhead)
            # - Large batches: diminishing returns (memory transfers dominate)
            
            # Efficiency curve: starts low, peaks at medium sizes, then decreases
            efficiency = 1.0 - 0.7 * (np.abs(batch_size - 16) / 16)
            
            # Base time plus size-dependent component
            base_time = 0.05
            size_factor = batch_size / efficiency
            noise = np.random.normal(0, 0.01)  # Add some noise
            
            processing_time = base_time + 0.01 * size_factor + noise
            
            # Simulate GPU memory usage increasing with batch size
            if torch.cuda.is_available():
                # Simulate allocating tensors
                dummy_tensors = [torch.zeros(batch_size, 1024, 1024) for _ in range(3)]
                torch.cuda.synchronize()
                time.sleep(processing_time)
                del dummy_tensors
                gc.collect()
                torch.cuda.empty_cache()
            else:
                # Just sleep to simulate processing time
                time.sleep(processing_time)
            
            return [f"result_{i}" for i in range(batch_size)]
    else:
        # TODO: Replace with actual data processing function
        def process_batch(batch):
            # Process actual data
            # ...
            return batch
    
    # Apply monitoring decorator
    monitored_process = monitor_process_function(batch_sizer)(process_batch)
    
    try:
        # Process batches
        for i in range(num_iterations):
            # Generate a dummy batch of current size
            batch = [f"item_{j}" for j in range(batch_sizer.batch_size)]
            
            # Process the batch
            print(f"Iteration {i+1}/{num_iterations}: processing batch of size {len(batch)}")
            results = monitored_process(batch)
            
            # Periodically apply optimal batch size from monitoring data
            if (i+1) % apply_optimal_every == 0:
                batch_sizer.apply_optimal_batch_size()
                
            # Short pause between iterations
            time.sleep(0.1)
            
        # Generate final report
        report = batch_sizer.generate_performance_report("final_batch_performance.png")
        print("\nPerformance Summary:")
        for key, value in report.items():
            print(f"  {key}: {value}")
            
    except KeyboardInterrupt:
        print("\nExample stopped by user.")
    finally:
        # Stop monitoring
        batch_sizer.stop_monitoring()
        print("Example completed.")


if __name__ == "__main__":
    # Run the example
    run_example(
        num_iterations=50,
        initial_batch_size=4,
        simulated_processing=True,
        apply_optimal_every=10
    ) 