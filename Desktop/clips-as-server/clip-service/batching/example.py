#!/usr/bin/env python3
"""
Example usage of the DynamicBatcher class.
"""

import time
import torch
import numpy as np
import logging
import argparse
from torch.utils.data import DataLoader, TensorDataset, Dataset
from dynamic_batcher import DynamicBatcher, batch_processor

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Example model for demonstration
class SimpleModel(torch.nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# Example dataset that can generate different sizes of data
class DynamicShapeDataset(Dataset):
    def __init__(self, shapes=None, samples=1000):
        self.samples = samples
        if shapes is None:
            # Default shapes: small, medium, large
            self.shapes = [(128,), (512,), (2048,)]
        else:
            self.shapes = shapes
        
        # Pre-generate data once
        self.data = []
        for _ in range(samples):
            shape_idx = np.random.randint(0, len(self.shapes))
            shape = self.shapes[shape_idx]
            self.data.append((torch.randn(shape), torch.randn(128)))
    
    def __len__(self):
        return self.samples
    
    def __getitem__(self, idx):
        return self.data[idx]

def basic_example():
    """Basic usage of dynamic batcher"""
    # Initialize dynamic batcher
    batcher = DynamicBatcher(
        initial_batch_size=16,
        min_batch_size=4,
        max_batch_size=64,
        target_memory_usage=0.7,
        target_latency=0.1,
        enable_monitoring=True
    )
    
    # Create model and data
    model = SimpleModel().to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate some random input data
    data = torch.randn(1000, 512)
    
    # Define processing function
    @batch_processor(batcher)
    def process_batch(batch):
        # Simulate some processing time varying with batch size
        time.sleep(0.001 * len(batch))  
        with torch.no_grad():
            output = model(batch)
        return output.cpu().numpy()
    
    # Process all data
    logger.info("Processing data using batches...")
    outputs = process_batch(data)
    logger.info(f"Processed {len(outputs)} items")
    
    # Save and display metrics
    metrics_file = batcher.save_metrics()
    logger.info(f"Metrics saved to {metrics_file}")
    
    # Show average metrics
    avg_metrics = batcher.metrics.get_average_metrics()
    logger.info(f"Average metrics: {avg_metrics}")

def pytorch_dataloader_example():
    """Example integration with PyTorch DataLoader"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize dynamic batcher
    batcher = DynamicBatcher(
        initial_batch_size=16,
        min_batch_size=4,
        max_batch_size=128,
        target_memory_usage=0.7,
        auto_tune=True,
        device=device
    )
    
    # Create model
    model = SimpleModel().to(device)
    
    # Create dataset
    dataset = DynamicShapeDataset(samples=2000)
    
    # Function to create dataloader with current batch size
    def create_dataloader():
        return DataLoader(
            dataset,
            batch_size=batcher.get_batch_size(),
            num_workers=batcher.get_optimal_workers(),
            pin_memory=(device == 'cuda')
        )
    
    # Get initial dataloader
    dataloader = create_dataloader()
    
    # Process batches
    logger.info("Processing data using PyTorch DataLoader...")
    total_processed = 0
    batch_count = 0
    
    # Process all data
    for epoch in range(2):
        logger.info(f"Epoch {epoch+1}")
        
        for inputs, targets in dataloader:
            batch_size = inputs[0].shape[0]
            batch_count += 1
            
            # Move to device
            inputs = inputs[0].to(device)
            targets = targets.to(device)
            
            # Process batch and time it
            start_time = time.time()
            with torch.no_grad():
                outputs = model(inputs)
            processing_time = time.time() - start_time
            
            # Update total processed
            total_processed += batch_size
            
            # Adjust batch size based on performance
            prev_batch_size = batcher.get_batch_size()
            batcher.adjust_batch_size(
                processing_time=processing_time,
                batch_shape=inputs.shape
            )
            
            # Recreate dataloader if batch size changed
            if prev_batch_size != batcher.get_batch_size():
                dataloader = create_dataloader()
                logger.info(f"Recreated DataLoader with batch_size={batcher.get_batch_size()}")
        
        logger.info(f"Epoch {epoch+1} complete, processed {total_processed} samples")
    
    # Save metrics
    metrics_file = batcher.save_metrics("pytorch_dataloader_metrics.json")
    logger.info(f"Metrics saved to {metrics_file}")

def different_shapes_example():
    """Example showing how to optimize for different input shapes"""
    # Initialize dynamic batcher with auto-tuning
    batcher = DynamicBatcher(
        initial_batch_size=32,
        min_batch_size=1,
        max_batch_size=128,
        target_memory_usage=0.7,
        auto_tune=True,
        enable_monitoring=True
    )
    
    # Create model
    model = SimpleModel(input_dim=512).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Different input shapes to test
    shapes = [
        (128,),   # Small input
        (512,),   # Medium input
        (2048,),  # Large input
    ]
    
    # Test each shape
    for shape in shapes:
        input_dim = shape[0]
        logger.info(f"Testing input shape: {shape}")
        
        # Get optimal batch size for this shape
        optimal_batch_size = batcher.optimize_for_input_size(shape)
        logger.info(f"Optimal batch size for shape {shape}: {optimal_batch_size}")
        
        # Generate data for this shape
        data = torch.randn(1000, input_dim)
        
        # Define processing function for this shape
        @batch_processor(batcher)
        def process_batch(batch):
            with torch.no_grad():
                output = model(batch)
            return output.cpu().numpy()
        
        # Process all data
        outputs = process_batch(data)
        logger.info(f"Processed {len(outputs)} items with shape {shape}")
    
    # Save metrics
    metrics_file = batcher.save_metrics("shape_optimization_metrics.json")
    logger.info(f"Metrics saved to {metrics_file}")

def main():
    parser = argparse.ArgumentParser(description='Dynamic batcher examples')
    parser.add_argument('--example', type=str, default='basic',
                        choices=['basic', 'pytorch', 'shapes', 'all'],
                        help='Which example to run')
    
    args = parser.parse_args()
    
    if args.example == 'basic' or args.example == 'all':
        logger.info("Running basic example")
        basic_example()
    
    if args.example == 'pytorch' or args.example == 'all':
        logger.info("Running PyTorch DataLoader example")
        pytorch_dataloader_example()
    
    if args.example == 'shapes' or args.example == 'all':
        logger.info("Running different shapes example")
        different_shapes_example()

if __name__ == "__main__":
    main() 