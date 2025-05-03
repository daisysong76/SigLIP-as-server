#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dynamic Batch Sizing Example for CLIP Service

This example demonstrates how to integrate the DynamicBatcher with a PyTorch
CLIP model for optimal performance.
"""

import sys
import os
import time
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import clip
from PIL import Image
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our dynamic batcher
from batching.dynamic_batcher import DynamicBatcher, batch_processor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageTextDataset(Dataset):
    """
    Simple dataset of image-text pairs for demonstration purposes.
    In a real-world scenario, this would be your production dataset.
    """
    def __init__(self, folder_path, max_samples=1000):
        self.folder_path = Path(folder_path)
        self.image_files = list(self.folder_path.glob("*.jpg"))[:max_samples]
        self.preprocess = None  # Will be set later based on model
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        # Use filename as caption for demo purposes
        text = image_path.stem.replace("_", " ")
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        if self.preprocess:
            image = self.preprocess(image)
            
        return {
            "image": image,
            "text": text,
            "image_path": str(image_path)
        }


class DynamicClipProcessor:
    """
    CLIP processor with dynamic batch sizing for optimal throughput.
    """
    def __init__(
        self,
        model_name="ViT-B/32",
        initial_batch_size=16,
        min_batch_size=1,
        max_batch_size=64,
        device=None,
        enable_monitoring=True
    ):
        # Use specified device or auto-detect
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load CLIP model
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        logger.info(f"Loaded CLIP model: {model_name}")
        
        # Initialize dynamic batcher
        self.batcher = DynamicBatcher(
            initial_batch_size=initial_batch_size,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            enable_monitoring=enable_monitoring
        )
        
        # Put model in evaluation mode
        self.model.eval()
    
    def get_optimal_batch_size(self):
        """Get the current optimal batch size"""
        return self.batcher.get_batch_size()
    
    def get_optimal_workers(self):
        """Get the optimal number of workers for data loading"""
        return self.batcher.get_optimal_workers()
    
    @torch.no_grad()
    def process_batch(self, batch):
        """
        Process a batch of data through CLIP model.
        
        Args:
            batch: Dictionary with 'image' and 'text' keys
            
        Returns:
            Dictionary with embeddings and similarity scores
        """
        images = batch["image"].to(self.device)
        texts = batch["text"]
        
        # Measure batch processing time
        start_time = time.time()
        
        # Get text features
        text_tokens = clip.tokenize(texts).to(self.device)
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(text_tokens)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # Move results back to CPU
        image_features = image_features.cpu()
        text_features = text_features.cpu()
        similarity = similarity.cpu()
        
        # Record processing time and adjust batch size
        processing_time = time.time() - start_time
        new_batch_size = self.batcher.adjust_batch_size(
            processing_time=processing_time,
            batch_shape=images.shape
        )
        
        return {
            "image_features": image_features,
            "text_features": text_features,
            "similarity": similarity,
            "processing_time": processing_time,
            "batch_size": len(images),
            "next_batch_size": new_batch_size
        }
    
    def create_dataloader(self, dataset, shuffle=True):
        """
        Create a dataloader with optimal batch size and worker settings.
        
        Args:
            dataset: PyTorch dataset
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader configured with optimal settings
        """
        # Set the preprocess function if not already set
        if hasattr(dataset, "preprocess") and dataset.preprocess is None:
            dataset.preprocess = self.preprocess
        
        # Create DataLoader with dynamic batch size and workers
        loader = DataLoader(
            dataset,
            batch_size=self.get_optimal_batch_size(),
            shuffle=shuffle,
            num_workers=self.get_optimal_workers(),
            pin_memory=True if self.device == "cuda" else False
        )
        
        return loader


def run_clip_benchmark(image_folder, num_iterations=100):
    """
    Run a benchmark with the dynamic batcher applied to CLIP processing.
    
    Args:
        image_folder: Folder containing images for testing
        num_iterations: Number of iterations to run
    """
    # Create dataset and processor
    dataset = ImageTextDataset(image_folder)
    
    # Display dataset info
    logger.info(f"Dataset size: {len(dataset)} images")
    
    # Initialize processor with dynamic batching
    processor = DynamicClipProcessor(
        model_name="ViT-B/32",
        initial_batch_size=8,  # Start conservative
        min_batch_size=1,
        max_batch_size=128,
        enable_monitoring=True
    )
    
    # Create dataloader with initial settings
    dataloader = processor.create_dataloader(dataset)
    
    # Tracking variables
    total_images = 0
    total_time = 0
    batch_count = 0
    
    # Run benchmark for specified iterations or until dataset exhausted
    logger.info(f"Starting benchmark with dynamic batch sizing")
    start_time = time.time()
    
    for iteration in range(num_iterations):
        # Recreate dataloader each iteration to apply new batch size
        if batch_count > 0:
            dataloader = processor.create_dataloader(dataset)
            
        # Process one batch
        for batch_idx, batch in enumerate(dataloader):
            # Process through CLIP
            result = processor.process_batch(batch)
            
            # Update counters
            batch_size = result["batch_size"]
            processing_time = result["processing_time"]
            total_images += batch_size
            total_time += processing_time
            batch_count += 1
            
            # Log progress
            if batch_count % 5 == 0:
                metrics = processor.batcher.get_metrics()
                logger.info(
                    f"Batch {batch_count}: size={batch_size}, "
                    f"time={processing_time:.3f}s, "
                    f"throughput={batch_size/processing_time:.1f} img/s, "
                    f"next_batch_size={result['next_batch_size']}, "
                    f"memory={metrics['memory_usage']:.1%}"
                )
            
            break  # Only process one batch per iteration
    
    # Get final metrics
    elapsed = time.time() - start_time
    overall_throughput = total_images / max(total_time, 0.001)
    
    # Log summary
    logger.info(f"Benchmark complete:")
    logger.info(f"Processed {total_images} images in {elapsed:.2f} seconds")
    logger.info(f"Average throughput: {overall_throughput:.2f} images/second")
    
    # Get metrics and save to file
    metrics_file = processor.batcher.save_metrics()
    logger.info(f"Metrics saved to: {metrics_file}")
    
    return processor.batcher.get_metrics()


if __name__ == "__main__":
    # Get image folder from arguments or use default
    import argparse
    parser = argparse.ArgumentParser(description="Run CLIP benchmark with dynamic batch sizing")
    parser.add_argument("--image_folder", type=str, default="./data/images", 
                        help="Folder containing images for benchmark")
    parser.add_argument("--iterations", type=int, default=20, 
                        help="Number of iterations to run")
    args = parser.parse_args()
    
    # Run benchmark
    run_clip_benchmark(args.image_folder, args.iterations) 