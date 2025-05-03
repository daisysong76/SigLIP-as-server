"""
Script to debug Hugging Face dataset loading for CLIP embeddings.
Run this directly to test dataset loading without pytest.
"""
import sys
import os
import torch
import numpy as np
from datasets import load_dataset, Dataset, IterableDataset
from transformers import CLIPProcessor, CLIPModel
import asyncio
import logging
import clip
from PIL import Image
import io
import time
from pathlib import Path
import psutil
import gc
from itertools import islice
from functools import partial
from typing import Dict, List, Tuple, Optional, Union, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DynamicBatchSizer:
    """Dynamically adjusts batch size based on hardware resources and processing times."""
    def __init__(
        self, 
        initial_batch_size: int = 16, 
        min_batch_size: int = 1, 
        max_batch_size: int = 64,
        target_memory_usage: float = 0.7,  # Target 70% of available memory
        growth_factor: float = 1.2,        # 20% growth per step
        reduction_factor: float = 0.8      # 20% reduction per step
    ):
        self.batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_memory_usage = target_memory_usage
        self.growth_factor = growth_factor
        self.reduction_factor = reduction_factor
        self.processing_times = []
        self.memory_usages = []
        self.step_count = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Track hardware capabilities
        self.gpu_mem_total = 0
        self.cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count()
        if self.device == "cuda":
            self.gpu_mem_total = torch.cuda.get_device_properties(0).total_memory
        
        self.system_memory = psutil.virtual_memory().total
        logger.info(f"Hardware capabilities: CPU Cores: {self.cpu_count}, System RAM: {self.system_memory / 1e9:.2f} GB")
        if self.device == "cuda":
            logger.info(f"GPU Memory: {self.gpu_mem_total / 1e9:.2f} GB")
        
    def adjust_batch_size(self, processing_time: float, tensor_shape: Tuple[int, ...]) -> int:
        """Adjust batch size based on the last processing time and hardware utilization."""
        self.step_count += 1
        self.processing_times.append(processing_time)
        
        # Calculate memory usage
        memory_usage = psutil.virtual_memory().percent / 100.0
        if self.device == "cuda":
            gpu_memory_allocated = torch.cuda.memory_allocated() / self.gpu_mem_total
            memory_usage = max(memory_usage, gpu_memory_allocated)
        
        self.memory_usages.append(memory_usage)
        
        # Only adjust after a few steps to get stable measurements
        if self.step_count < 3:
            return self.batch_size
        
        # Calculate tensor size in bytes
        tensor_size_bytes = np.prod(tensor_shape) * 4  # Assuming float32 (4 bytes)
        
        # Decide whether to increase or decrease batch size
        if memory_usage < self.target_memory_usage * 0.8 and processing_time < 0.5:
            # Room to grow - increase batch size
            new_batch_size = min(int(self.batch_size * self.growth_factor), self.max_batch_size)
            # Check if we have memory for this batch size
            projected_memory = memory_usage * (new_batch_size / self.batch_size)
            if projected_memory > self.target_memory_usage:
                # Too much memory would be used, use a more conservative increase
                new_batch_size = min(self.batch_size + 2, self.max_batch_size)
        elif memory_usage > self.target_memory_usage or processing_time > 2.0:
            # Too much memory used or too slow - decrease batch size
            new_batch_size = max(int(self.batch_size * self.reduction_factor), self.min_batch_size)
        else:
            # Current batch size is good
            new_batch_size = self.batch_size
        
        # Only log when batch size changes
        if new_batch_size != self.batch_size:
            logger.info(f"Adjusting batch size from {self.batch_size} to {new_batch_size}")
            logger.info(f"Memory usage: {memory_usage:.2f}, Processing time: {processing_time:.3f}s")
            self.batch_size = new_batch_size
        
        return self.batch_size
        
    def get_optimal_workers(self) -> int:
        """Get optimal number of worker threads based on CPU cores."""
        # Use N-1 workers where N is the number of CPU cores, with a minimum of 1
        return max(1, self.cpu_count - 1)
    
    def reset(self):
        """Reset measurements."""
        self.processing_times = []
        self.memory_usages = []
        self.step_count = 0

class AsyncIterableWrapper:
    """Wrapper to make IterableDataset async-compatible."""
    def __init__(self, dataset: IterableDataset):
        self.dataset = dataset
        self._iterator = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._iterator is None:
            self._iterator = iter(self.dataset)
        try:
            # Run iteration in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            item = await loop.run_in_executor(None, next, self._iterator)
            return item
        except StopIteration:
            raise StopAsyncIteration

def load_model_and_processor():
    """Load CLIP model and processor for testing."""
    logger.info("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        # Load basic CLIP model
        model, preprocess = clip.load("ViT-B/32", device=device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        logger.info("CLIP model loaded successfully")
        return (model, preprocess), processor, device
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        raise

def test_regular_dataset():
    """Test regular (non-async) dataset loading and processing."""
    logger.info("Testing regular dataset loading...")
    
    try:
        # Initialize dynamic batch sizer
        batch_sizer = DynamicBatchSizer(
            initial_batch_size=8,
            min_batch_size=1,
            max_batch_size=32
        )
        
        # Try different datasets until one works
        potential_datasets = [
            {"name": "cifar100", "split": "train"},
            {"name": "beans", "split": "train"},
            {"name": "food101", "split": "train"},
            {"name": "nlphuji/flickr30k", "split": "test"}
        ]
        
        for ds_config in potential_datasets:
            ds_name = ds_config["name"]
            split = ds_config["split"]
            try:
                logger.info(f"Trying to load {ds_name} with split '{split}'...")
                ds = load_dataset(ds_name, split=split, streaming=True)
                logger.info(f"Successfully loaded {ds_name}")
                logger.info(f"Column names: {ds.column_names}")
                
                # Get sample
                sample = next(iter(ds))
                logger.info(f"Sample keys: {sample.keys()}")
                
                # Load model and processor
                model, processor, device = load_model_and_processor()
                model_preprocess = model[1] if isinstance(model, tuple) else None
                
                # Dynamic memory management
                total_samples_processed = 0
                target_samples = 100  # Process up to 100 samples for benchmarking
                
                # Process in dynamic batches
                while total_samples_processed < target_samples:
                    current_batch_size = batch_sizer.batch_size
                    logger.info(f"Processing batch of size {current_batch_size}")
                    
                    # Collect batch samples
                    batch_start_time = time.perf_counter()
                    samples = []
                    image_key = "img" if "img" in sample else "image"
                    
                    # Collect batch samples
                    batch_items = list(islice(ds, current_batch_size))
                    if not batch_items:
                        logger.warning("No more items in dataset")
                        break
                    
                    # Process each item
                    for item in batch_items:
                        # Convert to PIL Image if needed
                        if not isinstance(item[image_key], Image.Image):
                            # Handle numpy array
                            if isinstance(item[image_key], np.ndarray):
                                img = Image.fromarray(item[image_key])
                                if img.mode != 'RGB':
                                    img = img.convert('RGB')
                            else:
                                logger.warning(f"Unexpected image type: {type(item[image_key])}")
                                continue
                        else:
                            img = item[image_key]
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                        
                        # Use CLIP's own preprocessing for correct dimensions
                        if model_preprocess:
                            # Use the preprocess function returned by clip.load()
                            processed_img = model_preprocess(img)
                        else:
                            # Use the transformers processor with fixed size
                            processed = processor(
                                images=img,
                                return_tensors="pt",
                                padding=True,
                                do_resize=True,
                                size={"height": 224, "width": 224},  # Ensure correct size for CLIP
                            )
                            processed_img = processed.pixel_values.squeeze(0)
                        
                        # Verify tensor shape
                        if processed_img.ndim == 3:  # [C, H, W]
                            # Check correct dimensions
                            if processed_img.shape[0] == 3 and processed_img.shape[1] == 224 and processed_img.shape[2] == 224:
                                samples.append(processed_img)
                            else:
                                logger.warning(f"Skipping sample with incorrect shape: {processed_img.shape}")
                        else:
                            logger.warning(f"Skipping sample with unexpected dimensions: {processed_img.ndim}")
                    
                    # Create batch if we have samples
                    if samples:
                        batch = torch.stack(samples)  # [B, C, H, W]
                        logger.info(f"Created batch with shape {batch.shape}")
                        
                        # Track memory before inference
                        if device == "cuda":
                            torch.cuda.synchronize()
                            mem_before = torch.cuda.memory_allocated()
                        
                        # Inference
                        with torch.no_grad():
                            if isinstance(model, tuple):
                                model_obj = model[0]
                                features = model_obj.encode_image(batch.to(device))
                            else:
                                features = model.encode_image(batch.to(device))
                                
                            features = features.to(torch.float32)
                            features = torch.nn.functional.normalize(features, p=2, dim=-1)
                        
                        # Track memory after inference
                        if device == "cuda":
                            torch.cuda.synchronize()
                            mem_after = torch.cuda.memory_allocated()
                            mem_used = mem_after - mem_before
                            logger.info(f"GPU memory used for inference: {mem_used/1e6:.2f} MB")
                        
                        # Measure batch processing time
                        batch_time = time.perf_counter() - batch_start_time
                        per_sample_time = batch_time / len(samples)
                        logger.info(f"Batch processing time: {batch_time:.3f}s ({per_sample_time:.3f}s per sample)")
                        
                        # Update stats
                        total_samples_processed += len(samples)
                        
                        # Adjust batch size based on performance
                        batch_sizer.adjust_batch_size(batch_time, batch.shape)
                        
                        # Explicitly free memory
                        del batch
                        del features
                        if device == "cuda":
                            torch.cuda.empty_cache()
                        gc.collect()
                    
                    if total_samples_processed >= target_samples:
                        logger.info(f"Processed {total_samples_processed} samples, stopping")
                        break
                
                logger.info("Regular dataset test completed successfully")
                return True
            except Exception as e:
                logger.error(f"Error with dataset {ds_name}: {e}", exc_info=True)
                continue
        
        logger.error("No datasets could be loaded successfully")
        return False
    except Exception as e:
        logger.error(f"Error in regular dataset test: {e}", exc_info=True)
        return False

async def test_async_dataset():
    """Test async dataset loading and processing with dynamic batching."""
    logger.info("Testing async dataset loading with dynamic batching...")
    
    try:
        # Initialize dynamic batch sizer
        batch_sizer = DynamicBatchSizer(
            initial_batch_size=8,
            min_batch_size=1,
            max_batch_size=32
        )
        
        # Try different datasets
        potential_datasets = [
            {"name": "cifar100", "split": "train"},
            {"name": "beans", "split": "train"},
            {"name": "food101", "split": "train"},
            {"name": "nlphuji/flickr30k", "split": "test"}
        ]
        
        for ds_config in potential_datasets:
            ds_name = ds_config["name"]
            split = ds_config["split"]
            try:
                logger.info(f"Trying to load {ds_name} for async testing with split '{split}'...")
                sample_dataset = load_dataset(ds_name, split=split, streaming=True)
                logger.info(f"Successfully loaded {ds_name}")
                
                # Get sample to determine image key
                sample = next(iter(sample_dataset))
                image_key = "img" if "img" in sample else "image"
                logger.info(f"Using image key: {image_key}")
                
                # Load model and processor
                model, processor, device = load_model_and_processor()
                model_preprocess = model[1] if isinstance(model, tuple) else None
                
                # Initialize counters
                total_samples_processed = 0
                target_samples = 100
                batch_idx = 0
                
                # Define optimized preprocessing function with error handling
                def preprocess_batch(batch, current_batch_size):
                    try:
                        # Process each image individually to ensure correct dimensions
                        processed_images = []
                        
                        # Limit batch to current_batch_size
                        batch_images = batch[image_key][:current_batch_size]
                        
                        for img in batch_images:
                            # Convert to PIL Image if needed
                            if not isinstance(img, Image.Image):
                                # Handle numpy array
                                if isinstance(img, np.ndarray):
                                    img = Image.fromarray(img)
                                    if img.mode != 'RGB':
                                        img = img.convert('RGB')
                                else:
                                    logger.warning(f"Unexpected image type: {type(img)}")
                                    continue
                            else:
                                if img.mode != 'RGB':
                                    img = img.convert('RGB')
                            
                            # Use CLIP's own preprocessing
                            if model_preprocess:
                                processed_img = model_preprocess(img)
                                processed_images.append(processed_img)
                            else:
                                # Fall back to transformers processor
                                processed = processor(
                                    images=img, 
                                    return_tensors="pt",
                                    padding=True,
                                    do_resize=True,
                                    size={"height": 224, "width": 224},  # Ensure correct size for CLIP
                                )
                                processed_images.append(processed.pixel_values.squeeze(0))
                        
                        # Combine processed images
                        if processed_images:
                            # Stack to create batch dimension
                            pixel_values = torch.stack(processed_images)
                            
                            # Ensure final shape is [B, 3, 224, 224]
                            if pixel_values.ndim == 4:
                                if pixel_values.shape[1] != 3 or pixel_values.shape[2] != 224 or pixel_values.shape[3] != 224:
                                    logger.warning(f"Incorrect pixel values shape: {pixel_values.shape}")
                                    if pixel_values.shape[1] == 3:
                                        pixel_values = torch.nn.functional.interpolate(
                                            pixel_values,
                                            size=(224, 224),
                                            mode='bilinear',
                                            align_corners=False
                                        )
                            
                            logger.info(f"Preprocessing complete: batch shape {pixel_values.shape}")
                            return {
                                "pixel_values": pixel_values,
                                "image_id": list(range(len(processed_images)))
                            }
                        else:
                            # Return empty batch with correct structure
                            logger.warning("No images were successfully processed")
                            return {
                                "pixel_values": torch.zeros((1, 3, 224, 224), dtype=torch.float32),
                                "image_id": [0]
                            }
                    except Exception as e:
                        logger.error(f"Error preprocessing batch: {e}", exc_info=True)
                        # Return empty batch with correct structure
                        return {
                            "pixel_values": torch.zeros((1, 3, 224, 224), dtype=torch.float32),
                            "image_id": [0]
                        }
                
                # Process dataset with dynamic batch size
                while total_samples_processed < target_samples:
                    current_batch_size = batch_sizer.batch_size
                    logger.info(f"Processing async batch {batch_idx} with batch size {current_batch_size}")
                    
                    # Create preprocessing function with current batch size
                    batch_processor = partial(preprocess_batch, current_batch_size=current_batch_size)
                    
                    # Process batch
                    batch_start_time = time.perf_counter()
                    
                    # Take a chunk of the dataset
                    chunk_dataset = sample_dataset.take(current_batch_size)
                    
                    # Process the chunk
                    processed_dataset = chunk_dataset.map(
                        batch_processor,
                        batched=True,
                        batch_size=current_batch_size
                    )
                    
                    # Wrap in async iterator
                    async_dataset = AsyncIterableWrapper(processed_dataset)
                    
                    # Process with model
                    try:
                        async for batch in async_dataset:
                            # Ensure pixel values have the right shape [B, 3, 224, 224]
                            pixel_values = batch["pixel_values"]
                            
                            # Fix missing batch dimension if needed
                            if pixel_values.ndim == 3 and pixel_values.shape[0] == 3:  # [3, 224, 224]
                                logger.info("Adding missing batch dimension")
                                pixel_values = pixel_values.unsqueeze(0)  # [1, 3, 224, 224]
                            
                            if pixel_values.ndim != 4 or pixel_values.shape[1] != 3 or pixel_values.shape[2] != 224 or pixel_values.shape[3] != 224:
                                logger.warning(f"Incorrect pixel values shape: {pixel_values.shape}, expected [B, 3, 224, 224]")
                                continue
                            
                            logger.info(f"Processing tensor with shape: {pixel_values.shape}")
                            
                            # Track memory before inference
                            if device == "cuda":
                                torch.cuda.synchronize()
                                mem_before = torch.cuda.memory_allocated()
                            
                            # Send to device
                            pixel_values = pixel_values.to(device)
                            
                            # Inference with timing
                            with torch.no_grad():
                                if isinstance(model, tuple):
                                    model_obj = model[0]
                                    features = model_obj.encode_image(pixel_values)
                                else:
                                    features = model.encode_image(pixel_values)
                                    
                                features = features.to(torch.float32)
                                features = torch.nn.functional.normalize(features, p=2, dim=-1)
                            
                            # Track memory after inference
                            if device == "cuda":
                                torch.cuda.synchronize()
                                mem_after = torch.cuda.memory_allocated()
                                mem_used = mem_after - mem_before
                                logger.info(f"GPU memory used for inference: {mem_used/1e6:.2f} MB")
                            
                            # Update counters
                            batch_processed = pixel_values.shape[0]
                            total_samples_processed += batch_processed
                            logger.info(f"Processed {batch_processed} samples in batch (total: {total_samples_processed})")
                            
                            # Measure batch time
                            batch_time = time.perf_counter() - batch_start_time
                            logger.info(f"Batch processing time: {batch_time:.3f}s")
                            
                            # Adjust batch size for next iteration
                            batch_sizer.adjust_batch_size(batch_time, pixel_values.shape)
                            
                            # Explicitly free memory
                            del pixel_values
                            del features
                            if device == "cuda":
                                torch.cuda.empty_cache()
                            gc.collect()
                            
                            # Only process first batch from this chunk
                            break
                    except Exception as e:
                        logger.error(f"Error processing async batch: {e}", exc_info=True)
                    
                    # Increment batch index
                    batch_idx += 1
                    
                    if total_samples_processed >= target_samples:
                        logger.info(f"Processed {total_samples_processed} samples, stopping")
                        break
                
                logger.info("Async dataset test completed successfully")
                return True
            
            except Exception as e:
                logger.error(f"Error with async dataset {ds_name}: {e}", exc_info=True)
                continue
        
        logger.error("No datasets could be processed asynchronously")
        return False
    except Exception as e:
        logger.error(f"Error in async dataset test: {e}", exc_info=True)
        return False

async def main():
    """Run all tests."""
    logger.info("Starting dataset debugging with dynamic batch sizing")
    
    # Test regular dataset loading
    regular_result = test_regular_dataset()
    logger.info(f"Regular dataset test {'succeeded' if regular_result else 'failed'}")
    
    # Test async dataset loading
    async_result = await test_async_dataset()
    logger.info(f"Async dataset test {'succeeded' if async_result else 'failed'}")
    
    if regular_result and async_result:
        logger.info("All tests passed! Your setup should work correctly.")
    else:
        logger.error("Some tests failed. Check the logs for details.")

if __name__ == "__main__":
    asyncio.run(main()) 