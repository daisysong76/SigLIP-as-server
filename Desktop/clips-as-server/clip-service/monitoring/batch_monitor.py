"""
Batch size performance monitoring for CLIP service.
Tracks and visualizes batch size performance metrics in real-time.
"""

import time
import logging
import threading
import numpy as np
import psutil
import torch
from typing import Dict, List, Optional, Tuple, Union
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from datetime import datetime
import os
import json
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchPerformanceMonitor:
    """Monitors batch size performance metrics for the CLIP service."""
    
    def __init__(
        self,
        window_size: int = 100,
        enable_live_plot: bool = True,
        save_metrics: bool = True,
        metrics_dir: str = "metrics",
        plot_update_interval: float = 1.0,  # seconds
    ):
        self.window_size = window_size
        self.enable_live_plot = enable_live_plot
        self.save_metrics = save_metrics
        self.metrics_dir = Path(metrics_dir)
        self.plot_update_interval = plot_update_interval
        
        # Create metrics directory if it doesn't exist
        if self.save_metrics:
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.batch_sizes = deque(maxlen=window_size)
        self.processing_times = deque(maxlen=window_size)
        self.sample_throughputs = deque(maxlen=window_size)
        self.memory_usages = deque(maxlen=window_size)
        self.gpu_memory_usages = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
        # Statistics
        self.total_samples_processed = 0
        self.total_batches_processed = 0
        self.start_time = time.time()
        
        # Live plotting
        self.fig = None
        self.animation = None
        self.plot_thread = None
        
        # Device info
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_mem_total = 0
        if self.device == "cuda":
            self.gpu_mem_total = torch.cuda.get_device_properties(0).total_memory
        
        logger.info(f"BatchPerformanceMonitor initialized with device={self.device}")
        
    def record_batch(
        self,
        batch_size: int,
        processing_time: float,
        memory_usage: Optional[float] = None,
        gpu_memory_usage: Optional[float] = None
    ):
        """Record metrics for a processed batch."""
        timestamp = time.time()
        
        # Calculate throughput (samples/second)
        throughput = batch_size / processing_time if processing_time > 0 else 0
        
        # Measure memory usage if not provided
        if memory_usage is None:
            memory_usage = psutil.virtual_memory().percent / 100.0
            
        # Measure GPU memory if CUDA available and not provided
        if gpu_memory_usage is None and self.device == "cuda":
            gpu_memory_usage = torch.cuda.memory_allocated() / self.gpu_mem_total
        
        # Store metrics
        self.batch_sizes.append(batch_size)
        self.processing_times.append(processing_time)
        self.sample_throughputs.append(throughput)
        self.memory_usages.append(memory_usage)
        if gpu_memory_usage is not None:
            self.gpu_memory_usages.append(gpu_memory_usage)
        self.timestamps.append(timestamp)
        
        # Update statistics
        self.total_samples_processed += batch_size
        self.total_batches_processed += 1
        
        # Log metrics
        if self.total_batches_processed % 10 == 0:
            avg_throughput = self.get_average_throughput()
            logger.info(
                f"Batch #{self.total_batches_processed}: size={batch_size}, "
                f"time={processing_time:.4f}s, throughput={throughput:.2f} samples/s, "
                f"avg_throughput={avg_throughput:.2f} samples/s"
            )
        
        # Save metrics if enabled
        if self.save_metrics and self.total_batches_processed % 10 == 0:
            self.save_metrics_to_file()
            
    def get_average_throughput(self, window: int = None) -> float:
        """Get average throughput over the specified window (samples/second)."""
        if not self.sample_throughputs:
            return 0.0
            
        if window is None or window >= len(self.sample_throughputs):
            return np.mean(self.sample_throughputs)
        else:
            return np.mean(list(self.sample_throughputs)[-window:])
            
    def get_average_batch_size(self, window: int = None) -> float:
        """Get average batch size over the specified window."""
        if not self.batch_sizes:
            return 0.0
            
        if window is None or window >= len(self.batch_sizes):
            return np.mean(self.batch_sizes)
        else:
            return np.mean(list(self.batch_sizes)[-window:])
    
    def get_optimal_batch_size(self) -> int:
        """Estimate the optimal batch size based on performance metrics."""
        if len(self.batch_sizes) < 5:
            return max(self.batch_sizes) if self.batch_sizes else 16
            
        # Create arrays from deques
        batch_sizes = np.array(self.batch_sizes)
        throughputs = np.array(self.sample_throughputs)
        
        # Group by batch size
        unique_sizes = np.unique(batch_sizes)
        avg_throughputs = []
        
        for size in unique_sizes:
            indices = np.where(batch_sizes == size)
            avg_throughput = np.mean(throughputs[indices])
            avg_throughputs.append((size, avg_throughput))
        
        # Sort by throughput
        avg_throughputs.sort(key=lambda x: x[1], reverse=True)
        
        # Return batch size with highest average throughput
        return int(avg_throughputs[0][0]) if avg_throughputs else 16
    
    def get_performance_summary(self) -> Dict:
        """Get a summary of performance metrics."""
        elapsed_time = time.time() - self.start_time
        
        return {
            "total_samples": self.total_samples_processed,
            "total_batches": self.total_batches_processed,
            "elapsed_time": elapsed_time,
            "overall_throughput": self.total_samples_processed / elapsed_time if elapsed_time > 0 else 0,
            "avg_batch_size": self.get_average_batch_size(),
            "avg_throughput": self.get_average_throughput(),
            "optimal_batch_size": self.get_optimal_batch_size(),
            "avg_processing_time": np.mean(self.processing_times) if self.processing_times else 0,
            "avg_memory_usage": np.mean(self.memory_usages) if self.memory_usages else 0,
            "avg_gpu_memory_usage": np.mean(self.gpu_memory_usages) if self.gpu_memory_usages else 0,
            "device": self.device,
            "timestamp": datetime.now().isoformat()
        }
        
    def save_metrics_to_file(self):
        """Save performance metrics to a JSON file."""
        if not self.save_metrics:
            return
            
        summary = self.get_performance_summary()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.metrics_dir / f"batch_metrics_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.debug(f"Saved metrics to {filename}")
    
    def start_live_plot(self):
        """Start a live plot of performance metrics in a separate thread."""
        if not self.enable_live_plot:
            return
            
        def run_plot():
            # Set up the plot
            plt.style.use('ggplot')
            self.fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            self.fig.suptitle("CLIP Service Batch Performance Monitor", fontsize=16)
            
            # Flatten axes for easier indexing
            axes = axes.flatten()
            
            # Throughput and batch size plot
            throughput_ax = axes[0]
            batch_size_ax = throughput_ax.twinx()
            throughput_line, = throughput_ax.plot([], [], 'b-', label='Throughput')
            batch_size_line, = batch_size_ax.plot([], [], 'r-', label='Batch Size')
            throughput_ax.set_ylabel('Throughput (samples/s)', color='b')
            batch_size_ax.set_ylabel('Batch Size', color='r')
            throughput_ax.tick_params(axis='y', labelcolor='b')
            batch_size_ax.tick_params(axis='y', labelcolor='r')
            
            # Add combined legend
            lines = [throughput_line, batch_size_line]
            throughput_ax.legend(lines, [line.get_label() for line in lines])
            
            # Processing time plot
            time_ax = axes[1]
            time_line, = time_ax.plot([], [], 'g-')
            time_ax.set_ylabel('Processing Time (s)')
            time_ax.set_title('Batch Processing Time')
            
            # Memory usage plot
            memory_ax = axes[2]
            memory_line, = memory_ax.plot([], [], 'm-')
            memory_ax.set_ylabel('Memory Usage (%)')
            memory_ax.set_title('System Memory Usage')
            memory_ax.set_ylim(0, 100)
            
            # Throughput vs Batch Size scatter plot
            scatter_ax = axes[3]
            scatter = scatter_ax.scatter([], [], c=[], cmap='viridis')
            scatter_ax.set_xlabel('Batch Size')
            scatter_ax.set_ylabel('Throughput (samples/s)')
            scatter_ax.set_title('Throughput vs Batch Size')
            
            # Initial X limits
            for ax in axes:
                ax.set_xlim(0, 10)
                
            # Function to update the plots
            def update(frame):
                if not self.timestamps:
                    return lines + [scatter]
                    
                # Convert deques to lists for plotting
                timestamps = np.array(self.timestamps)
                rel_timestamps = timestamps - timestamps[0]  # Relative time from start
                throughputs = np.array(self.sample_throughputs)
                batch_sizes = np.array(self.batch_sizes)
                proc_times = np.array(self.processing_times)
                mem_usages = np.array(self.memory_usages) * 100  # Convert to percentage
                
                # Update throughput and batch size plot
                throughput_line.set_data(range(len(throughputs)), throughputs)
                batch_size_line.set_data(range(len(batch_sizes)), batch_sizes)
                throughput_ax.set_xlim(0, len(throughputs))
                if throughputs.size > 0:
                    throughput_ax.set_ylim(0, max(throughputs) * 1.1)
                if batch_sizes.size > 0:
                    batch_size_ax.set_ylim(0, max(batch_sizes) * 1.1)
                throughput_ax.set_title(f'Throughput and Batch Size (Avg: {self.get_average_throughput():.2f} samples/s)')
                
                # Update processing time plot
                time_line.set_data(range(len(proc_times)), proc_times)
                time_ax.set_xlim(0, len(proc_times))
                if proc_times.size > 0:
                    time_ax.set_ylim(0, max(proc_times) * 1.1)
                
                # Update memory usage plot
                memory_line.set_data(range(len(mem_usages)), mem_usages)
                memory_ax.set_xlim(0, len(mem_usages))
                
                # Update scatter plot
                if len(batch_sizes) > 0 and len(throughputs) > 0:
                    scatter.set_offsets(np.c_[batch_sizes, throughputs])
                    scatter.set_array(np.arange(len(batch_sizes)))
                    scatter_ax.set_xlim(min(batch_sizes) * 0.9, max(batch_sizes) * 1.1)
                    scatter_ax.set_ylim(0, max(throughputs) * 1.1)
                    
                    # Add optimal batch size annotation
                    optimal = self.get_optimal_batch_size()
                    scatter_ax.set_title(f'Throughput vs Batch Size (Optimal: {optimal})')
                
                return lines + [scatter]
            
            # Create animation
            self.animation = FuncAnimation(
                self.fig, update, interval=self.plot_update_interval * 1000,
                blit=True, cache_frame_data=False
            )
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
        
        # Start plot in a separate thread
        self.plot_thread = threading.Thread(target=run_plot)
        self.plot_thread.daemon = True
        self.plot_thread.start()
        
        logger.info("Live plotting started in background thread")
    
    def generate_report(self, save_path: str = None) -> Dict:
        """Generate a comprehensive performance report with visualizations."""
        if not self.batch_sizes:
            logger.warning("No data available to generate report")
            return {"error": "No data available"}
            
        # Generate report data
        summary = self.get_performance_summary()
        
        # Create report figure
        plt.style.use('ggplot')
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle("CLIP Service Batch Performance Report", fontsize=16)
        
        # Convert deques to arrays
        timestamps = np.array(self.timestamps)
        rel_timestamps = timestamps - timestamps[0]  # Relative time from start
        throughputs = np.array(self.sample_throughputs)
        batch_sizes = np.array(self.batch_sizes)
        proc_times = np.array(self.processing_times)
        mem_usages = np.array(self.memory_usages) * 100  # Convert to percentage
        
        # 1. Throughput over time
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(rel_timestamps, throughputs, 'b-', label='Throughput')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Throughput (samples/s)')
        ax1.set_title(f'Throughput Over Time (Avg: {summary["avg_throughput"]:.2f} samples/s)')
        ax1.grid(True)
        
        # 2. Batch size evolution
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(rel_timestamps, batch_sizes, 'r-', label='Batch Size')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Batch Size')
        ax2.set_title(f'Batch Size Evolution (Avg: {summary["avg_batch_size"]:.1f})')
        ax2.grid(True)
        
        # 3. Throughput vs Batch Size scatter
        ax3 = plt.subplot(2, 2, 3)
        scatter = ax3.scatter(batch_sizes, throughputs, c=rel_timestamps, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, ax=ax3, label='Time (s)')
        
        # Add trendline
        if len(batch_sizes) > 1:
            z = np.polyfit(batch_sizes, throughputs, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(batch_sizes), max(batch_sizes), 100)
            ax3.plot(x_range, p(x_range), "r--", alpha=0.8)
            
        ax3.set_xlabel('Batch Size')
        ax3.set_ylabel('Throughput (samples/s)')
        ax3.set_title(f'Throughput vs Batch Size (Optimal: {summary["optimal_batch_size"]})')
        ax3.grid(True)
        
        # 4. Processing time distribution
        ax4 = plt.subplot(2, 2, 4)
        sns.histplot(proc_times, kde=True, ax=ax4)
        ax4.set_xlabel('Processing Time (s)')
        ax4.set_ylabel('Frequency')
        ax4.set_title(f'Processing Time Distribution (Avg: {summary["avg_processing_time"]:.4f}s)')
        
        # Add summary text
        summary_text = (
            f"Total samples processed: {summary['total_samples']}\n"
            f"Total batches processed: {summary['total_batches']}\n"
            f"Elapsed time: {summary['elapsed_time']:.2f}s\n"
            f"Overall throughput: {summary['overall_throughput']:.2f} samples/s\n"
            f"Optimal batch size: {summary['optimal_batch_size']}\n"
            f"Device: {summary['device']}"
        )
        
        fig.text(0.5, 0.01, summary_text, ha='center', fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.8))
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(0.98, 0.01, f"Generated: {timestamp}", ha='right', fontsize=8)
        
        # Adjust layout and save if requested
        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Saved performance report to {save_path}")
        
        return summary
    
    def stop(self):
        """Stop monitoring and save final metrics."""
        if self.save_metrics:
            self.save_metrics_to_file()
            
        # Generate final report
        if self.batch_sizes:
            report_path = self.metrics_dir / "final_performance_report.png"
            self.generate_report(str(report_path))
            
        logger.info("Batch performance monitoring stopped")


# Decorator for easy monitoring
def monitor_batch(monitor: BatchPerformanceMonitor):
    """Decorator to automatically monitor batch processing functions."""
    def decorator(func):
        def wrapper(batch, *args, **kwargs):
            batch_size = len(batch) if hasattr(batch, '__len__') else 1
            start_time = time.time()
            
            # Execute the original function
            result = func(batch, *args, **kwargs)
            
            # Record metrics
            processing_time = time.time() - start_time
            monitor.record_batch(batch_size, processing_time)
            
            return result
        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage
    print("Starting batch performance monitoring example...")
    
    # Initialize monitor
    monitor = BatchPerformanceMonitor(
        window_size=100,
        enable_live_plot=True,
        save_metrics=True,
        metrics_dir="metrics",
    )
    
    # Start live plotting
    monitor.start_live_plot()
    
    # Simulate batch processing with different batch sizes
    try:
        for i in range(50):
            # Simulate different batch sizes
            if i < 10:
                batch_size = 4
            elif i < 20:
                batch_size = 8
            elif i < 30:
                batch_size = 16
            elif i < 40:
                batch_size = 32
            else:
                batch_size = 64
                
            # Simulate processing time (larger batches take longer, but with diminishing returns)
            base_time = 0.1
            noise = np.random.normal(0, 0.02)
            processing_time = base_time * np.sqrt(batch_size / 8) + noise
            
            # Simulate memory usage (increases with batch size)
            memory_usage = 0.3 + (batch_size / 128)
            
            # Record batch metrics
            monitor.record_batch(batch_size, processing_time, memory_usage)
            
            # Sleep to simulate processing time
            time.sleep(0.2)
            
        # Generate and save final report
        monitor.generate_report("batch_monitoring_example.png")
        
        print("Example completed. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Example stopped by user.")
    finally:
        monitor.stop() 