import torch
import numpy as np
import time
import json
import os
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# Import our dynamic batcher
from dynamic_batcher import DynamicBatcher, batch_processor

# Configure logging
import logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BenchmarkModel(torch.nn.Module):
    """Simple model for benchmarking with configurable compute intensity"""
    def __init__(self, input_dim=512, hidden_dims=[1024, 512, 256], output_dim=128, 
                 compute_intensity=1.0):
        super().__init__()
        self.compute_intensity = compute_intensity
        
        # Create network with variable depth based on compute_intensity
        layers = []
        in_dim = input_dim
        
        # Adjust number of layers based on compute intensity
        num_extra_layers = int(compute_intensity * 3)
        
        for i, h_dim in enumerate(hidden_dims):
            layers.append(torch.nn.Linear(in_dim, h_dim))
            layers.append(torch.nn.ReLU())
            
            # Add extra layers based on compute intensity
            if i == len(hidden_dims) // 2:
                for _ in range(num_extra_layers):
                    layers.append(torch.nn.Linear(h_dim, h_dim))
                    layers.append(torch.nn.ReLU())
            
            in_dim = h_dim
        
        # Output layer
        layers.append(torch.nn.Linear(in_dim, output_dim))
        
        self.net = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        # Simulate variable computation time
        return self.net(x)


class BenchmarkDataset(Dataset):
    """Dataset with configurable sizes and memory requirements"""
    def __init__(self, samples=1000, input_dim=512, output_dim=128, 
                 variable_shapes=False, shape_range=(128, 4096)):
        self.samples = samples
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.variable_shapes = variable_shapes
        self.shape_range = shape_range
        
        # Pre-generate data
        self.data = []
        for _ in range(samples):
            if variable_shapes:
                # Random dimension between shape_range
                dim = np.random.randint(shape_range[0], shape_range[1])
                shape = (dim,)
            else:
                shape = (input_dim,)
                
            self.data.append((torch.randn(shape), torch.randn(output_dim)))
    
    def __len__(self):
        return self.samples
    
    def __getitem__(self, idx):
        return self.data[idx]


class BenchmarkRunner:
    """Benchmark different batching strategies"""
    def __init__(self, 
                 dataset_size=10000,
                 input_dim=512, 
                 output_dim=128,
                 compute_intensity=1.0,
                 variable_shapes=False,
                 device='cuda',
                 results_dir='benchmark_results'):
        
        self.dataset_size = dataset_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.compute_intensity = compute_intensity
        self.variable_shapes = variable_shapes
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Create model
        self.model = BenchmarkModel(
            input_dim=input_dim, 
            output_dim=output_dim,
            compute_intensity=compute_intensity
        ).to(self.device)
        
        # Create dataset
        self.dataset = BenchmarkDataset(
            samples=dataset_size,
            input_dim=input_dim,
            output_dim=output_dim,
            variable_shapes=variable_shapes
        )
        
        # Results
        self.results = {}
    
    def _create_dataloader(self, batch_size, num_workers=2, pin_memory=True):
        """Create a DataLoader with specified batch size"""
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory and self.device == 'cuda',
            shuffle=False
        )
    
    def benchmark_static_batch_sizes(self, batch_sizes=[1, 4, 8, 16, 32, 64, 128, 256]):
        """Benchmark model with static batch sizes"""
        results = {}
        
        for batch_size in tqdm(batch_sizes, desc="Testing static batch sizes"):
            # Create dataloader with this batch size
            dataloader = self._create_dataloader(batch_size)
            
            # Run benchmark
            start_time = time.time()
            total_items = 0
            batch_times = []
            memory_usages = []
            
            # Process all batches
            with torch.no_grad():
                for i, (inputs, _) in enumerate(dataloader):
                    # Move to device
                    if self.variable_shapes:
                        inputs = inputs.to(self.device)
                    else:
                        inputs = inputs.to(self.device)
                    
                    # Record memory before processing
                    if torch.cuda.is_available() and self.device == 'cuda':
                        torch.cuda.synchronize()
                        memory_before = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                    
                    # Process batch
                    batch_start = time.time()
                    outputs = self.model(inputs)
                    if torch.cuda.is_available() and self.device == 'cuda':
                        torch.cuda.synchronize()
                    batch_time = time.time() - batch_start
                    
                    # Record memory after processing
                    if torch.cuda.is_available() and self.device == 'cuda':
                        memory_after = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                        memory_usages.append(memory_after - memory_before)
                    
                    # Update counters
                    total_items += len(inputs)
                    batch_times.append(batch_time)
            
            # Calculate metrics
            total_time = time.time() - start_time
            throughput = total_items / total_time
            avg_batch_time = sum(batch_times) / len(batch_times)
            avg_memory_usage = sum(memory_usages) / len(memory_usages) if memory_usages else 0
            
            # Store results
            results[batch_size] = {
                'total_time': total_time,
                'throughput': throughput,
                'avg_batch_time': avg_batch_time,
                'avg_memory_usage': avg_memory_usage
            }
            
            logger.info(f"Batch size {batch_size}: Throughput = {throughput:.2f} items/s, "
                        f"Avg batch time = {avg_batch_time:.4f}s")
        
        self.results['static'] = results
        return results
    
    def benchmark_dynamic_batcher(self,
                                 initial_batch_size=16,
                                 min_batch_size=1,
                                 max_batch_size=256,
                                 target_memory_usage=0.7,
                                 target_latency=None,
                                 auto_tune=True):
        """Benchmark with dynamic batcher"""
        logger.info("Benchmarking dynamic batcher...")
        
        # Initialize dynamic batcher
        batcher = DynamicBatcher(
            initial_batch_size=initial_batch_size,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            target_memory_usage=target_memory_usage,
            target_latency=target_latency,
            auto_tune=auto_tune,
            enable_monitoring=True,
            device=self.device
        )
        
        # Create initial dataloader
        dataloader = self._create_dataloader(
            batch_size=batcher.get_batch_size(),
            num_workers=batcher.get_optimal_workers()
        )
        
        # Run benchmark
        start_time = time.time()
        total_items = 0
        batch_times = []
        batch_sizes = []
        memory_usages = []
        
        # Process all batches
        with torch.no_grad():
            batch_count = 0
            
            for epoch in range(2):  # Run 2 epochs to see how batch size stabilizes
                logger.info(f"Starting epoch {epoch+1}")
                
                for inputs, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                    batch_count += 1
                    batch_size = len(inputs)
                    batch_sizes.append(batch_size)
                    
                    # Move to device
                    if self.variable_shapes:
                        inputs = inputs.to(self.device)
                    else:
                        inputs = inputs.to(self.device)
                    
                    # Record memory before processing
                    if torch.cuda.is_available() and self.device == 'cuda':
                        torch.cuda.synchronize()
                        memory_before = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                    
                    # Process batch and time it
                    batch_start = time.time()
                    outputs = self.model(inputs)
                    if torch.cuda.is_available() and self.device == 'cuda':
                        torch.cuda.synchronize()
                    batch_time = time.time() - batch_start
                    batch_times.append(batch_time)
                    
                    # Record memory after processing
                    if torch.cuda.is_available() and self.device == 'cuda':
                        memory_after = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                        memory_usage = memory_after - memory_before
                        memory_usages.append(memory_usage)
                    else:
                        memory_usage = 0
                    
                    # Update batch size
                    prev_batch_size = batcher.get_batch_size()
                    batcher.adjust_batch_size(
                        processing_time=batch_time,
                        batch_shape=inputs.shape,
                        memory_usage=memory_usage
                    )
                    
                    # Recreate dataloader if batch size changed
                    if prev_batch_size != batcher.get_batch_size() and batch_count % 5 == 0:  # Only recreate every 5 batches to avoid overhead
                        dataloader = self._create_dataloader(
                            batch_size=batcher.get_batch_size(),
                            num_workers=batcher.get_optimal_workers()
                        )
                        logger.info(f"Adjusted batch size to {batcher.get_batch_size()}")
                    
                    # Update counters
                    total_items += batch_size
        
        # Calculate metrics
        total_time = time.time() - start_time
        throughput = total_items / total_time
        avg_batch_time = sum(batch_times) / len(batch_times)
        avg_batch_size = sum(batch_sizes) / len(batch_sizes)
        avg_memory_usage = sum(memory_usages) / len(memory_usages) if memory_usages else 0
        final_batch_size = batcher.get_batch_size()
        
        # Save batcher metrics
        metrics_file = batcher.save_metrics(os.path.join(self.results_dir, "dynamic_batcher_metrics.json"))
        
        # Store results
        results = {
            'total_time': total_time,
            'throughput': throughput,
            'avg_batch_time': avg_batch_time,
            'avg_batch_size': avg_batch_size,
            'avg_memory_usage': avg_memory_usage,
            'final_batch_size': final_batch_size,
            'batch_sizes': batch_sizes,
            'batch_times': batch_times,
            'memory_usages': memory_usages
        }
        
        logger.info(f"Dynamic batcher: Throughput = {throughput:.2f} items/s, "
                    f"Avg batch size = {avg_batch_size:.2f}, "
                    f"Final batch size = {final_batch_size}")
        
        self.results['dynamic'] = results
        return results
    
    def save_results(self, filename="benchmark_results.json"):
        """Save benchmark results to file"""
        path = self.results_dir / filename
        
        # Extract only serializable data
        serializable_results = {}
        
        # Process static results
        if 'static' in self.results:
            serializable_results['static'] = {}
            for batch_size, metrics in self.results['static'].items():
                serializable_results['static'][str(batch_size)] = metrics
        
        # Process dynamic results
        if 'dynamic' in self.results:
            serializable_results['dynamic'] = {
                k: v for k, v in self.results['dynamic'].items() 
                if k not in ['batch_sizes', 'batch_times', 'memory_usages']
            }
            # Store the batch sizes history in a separate file
            if 'batch_sizes' in self.results['dynamic']:
                np.save(
                    self.results_dir / "dynamic_batch_sizes.npy", 
                    np.array(self.results['dynamic']['batch_sizes'])
                )
            if 'batch_times' in self.results['dynamic']:
                np.save(
                    self.results_dir / "dynamic_batch_times.npy", 
                    np.array(self.results['dynamic']['batch_times'])
                )
            if 'memory_usages' in self.results['dynamic']:
                np.save(
                    self.results_dir / "dynamic_memory_usages.npy", 
                    np.array(self.results['dynamic']['memory_usages'])
                )
        
        # Save to file
        with open(path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        logger.info(f"Results saved to {path}")
        return path
    
    def plot_results(self, show_plots=True, save_plots=True):
        """Plot benchmark results"""
        if not self.results:
            logger.warning("No results to plot")
            return
        
        # Create figure directory
        figures_dir = self.results_dir / "figures"
        figures_dir.mkdir(exist_ok=True, parents=True)
        
        # 1. Plot throughput comparison
        if 'static' in self.results:
            # Get static throughputs
            batch_sizes = sorted([int(bs) for bs in self.results['static'].keys()])
            throughputs = [self.results['static'][bs]['throughput'] for bs in batch_sizes]
            
            plt.figure(figsize=(10, 6))
            plt.plot(batch_sizes, throughputs, 'bo-', label='Static batch size')
            
            # Add dynamic batcher result if available
            if 'dynamic' in self.results:
                dynamic_throughput = self.results['dynamic']['throughput']
                dynamic_batch_size = self.results['dynamic']['final_batch_size']
                plt.axhline(y=dynamic_throughput, color='r', linestyle='--', 
                           label=f'Dynamic batcher ({dynamic_batch_size:.1f})')
            
            plt.xscale('log', base=2)
            plt.xlabel('Batch Size')
            plt.ylabel('Throughput (items/s)')
            plt.title('Throughput vs Batch Size')
            plt.grid(True)
            plt.legend()
            
            if save_plots:
                plt.savefig(figures_dir / "throughput_comparison.png", dpi=300)
            
            if show_plots:
                plt.show()
            else:
                plt.close()
        
        # 2. Plot batch size evolution for dynamic batcher
        if 'dynamic' in self.results and 'batch_sizes' in self.results['dynamic']:
            batch_sizes = self.results['dynamic']['batch_sizes']
            plt.figure(figsize=(10, 6))
            plt.plot(batch_sizes)
            plt.xlabel('Batch Index')
            plt.ylabel('Batch Size')
            plt.title('Dynamic Batch Size Evolution')
            plt.grid(True)
            
            if save_plots:
                plt.savefig(figures_dir / "dynamic_batch_size_evolution.png", dpi=300)
            
            if show_plots:
                plt.show()
            else:
                plt.close()
        
        # 3. Plot batch time vs batch size
        if 'dynamic' in self.results and 'batch_sizes' in self.results['dynamic'] and 'batch_times' in self.results['dynamic']:
            batch_sizes = self.results['dynamic']['batch_sizes']
            batch_times = self.results['dynamic']['batch_times']
            
            plt.figure(figsize=(10, 6))
            plt.scatter(batch_sizes, batch_times, alpha=0.5)
            plt.xlabel('Batch Size')
            plt.ylabel('Batch Time (s)')
            plt.title('Batch Time vs Batch Size')
            plt.grid(True)
            
            if save_plots:
                plt.savefig(figures_dir / "batch_time_vs_size.png", dpi=300)
            
            if show_plots:
                plt.show()
            else:
                plt.close()
        
        # 4. Plot memory usage if available
        if 'dynamic' in self.results and 'memory_usages' in self.results['dynamic']:
            memory_usages = self.results['dynamic']['memory_usages']
            plt.figure(figsize=(10, 6))
            plt.plot(memory_usages)
            plt.xlabel('Batch Index')
            plt.ylabel('Memory Usage (GB)')
            plt.title('Memory Usage Over Time')
            plt.grid(True)
            
            if save_plots:
                plt.savefig(figures_dir / "memory_usage.png", dpi=300)
            
            if show_plots:
                plt.show()
            else:
                plt.close()


def main():
    """Main function to run benchmark"""
    parser = argparse.ArgumentParser(description='Benchmark dynamic batching vs static batch sizes')
    
    # General parameters
    parser.add_argument('--dataset-size', type=int, default=5000, 
                        help='Number of samples in dataset')
    parser.add_argument('--input-dim', type=int, default=512,
                        help='Input dimension for model')
    parser.add_argument('--compute-intensity', type=float, default=1.0,
                        help='Compute intensity factor (1.0 is normal)')
    parser.add_argument('--variable-shapes', action='store_true',
                        help='Use variable input shapes')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to run on')
    parser.add_argument('--results-dir', type=str, default='benchmark_results',
                        help='Directory to save results')
    
    # Static batch sizes to test
    parser.add_argument('--batch-sizes', type=str, default='1,2,4,8,16,32,64,128,256',
                        help='Comma-separated list of batch sizes to test')
    
    # Dynamic batcher parameters
    parser.add_argument('--initial-batch-size', type=int, default=16,
                        help='Initial batch size for dynamic batcher')
    parser.add_argument('--min-batch-size', type=int, default=1,
                        help='Minimum batch size for dynamic batcher')
    parser.add_argument('--max-batch-size', type=int, default=512,
                        help='Maximum batch size for dynamic batcher')
    parser.add_argument('--target-memory', type=float, default=0.7,
                        help='Target memory usage (0.0-1.0)')
    
    # Benchmark modes
    parser.add_argument('--skip-static', action='store_true',
                        help='Skip static batch size benchmarks')
    parser.add_argument('--skip-dynamic', action='store_true',
                        help='Skip dynamic batcher benchmark')
    parser.add_argument('--show-plots', action='store_true',
                        help='Show plots during benchmark')
    
    args = parser.parse_args()
    
    # Parse batch sizes
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(',')]
    
    # Create benchmark runner
    runner = BenchmarkRunner(
        dataset_size=args.dataset_size,
        input_dim=args.input_dim,
        compute_intensity=args.compute_intensity,
        variable_shapes=args.variable_shapes,
        device=args.device,
        results_dir=args.results_dir
    )
    
    # Run benchmarks
    if not args.skip_static:
        logger.info("Running static batch size benchmarks...")
        runner.benchmark_static_batch_sizes(batch_sizes=batch_sizes)
    
    if not args.skip_dynamic:
        logger.info("Running dynamic batcher benchmark...")
        runner.benchmark_dynamic_batcher(
            initial_batch_size=args.initial_batch_size,
            min_batch_size=args.min_batch_size,
            max_batch_size=args.max_batch_size,
            target_memory_usage=args.target_memory
        )
    
    # Save and plot results
    runner.save_results()
    runner.plot_results(show_plots=args.show_plots)
    
    logger.info("Benchmark complete!")


if __name__ == "__main__":
    main() 