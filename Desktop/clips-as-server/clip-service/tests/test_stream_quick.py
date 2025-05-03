"""
Quick performance test for CLIP image processing
"""

import sys
import os
import time
import logging
import asyncio
import gc
import psutil
import torch
import numpy as np
import clip
from transformers import CLIPProcessor
from pathlib import Path

# Add the parent directory to the path so we can import from clip-service
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from input.dataset_config import DatasetConfig
from inference.collate_fn import multimodal_collate_fn, postprocess_embeddings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

async def quick_test():
    """Run a quick performance test on image processing."""
    try:
        # Log system info
        mem = psutil.virtual_memory()
        logger.info(f"CPU cores: {psutil.cpu_count()}")
        logger.info(f"Memory available: {mem.available / (1024**3):.2f} GB")
        logger.info(f"Memory total: {mem.total / (1024**3):.2f} GB")
        
        # Load model
        logger.info("Loading CLIP model")
        model_start = time.perf_counter()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device, jit=True)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float32)
        logger.info(f"Model loaded in {time.perf_counter() - model_start:.2f}s on {device}")
        
        # Create smaller dataset config (faster to load)
        logger.info("Creating dataset config")
        ds_config = DatasetConfig(
            image_datasets=["cifar100"],  # Smaller dataset
            text_datasets=["cifar100"],
            batch_size=8,  # Smaller batch size
            max_text_length=77,
            image_size=224
        )
        
        # Load datasets
        logger.info("Loading datasets")
        dataset_start = time.perf_counter()
        image_dataset, text_dataset = await ds_config.load_streaming_dataset(
            processor,
            streaming=True
        )
        logger.info(f"Datasets loaded in {time.perf_counter() - dataset_start:.2f}s")
        
        # Process a few image batches to benchmark performance
        logger.info("Processing images")
        image_embeddings = []
        image_ids = []
        batch_count = 0
        batch_limit = 3  # Process only 3 batches
        
        async for batch in image_dataset:
            batch_start = time.perf_counter()
            logger.info(f"Batch {batch_count}: Processing...")
            
            # Check batch format
            logger.info(f"Batch keys: {batch.keys()}")
            pixel_values_shape = batch["pixel_values"].shape if "pixel_values" in batch else "N/A"
            logger.info(f"Pixel values shape: {pixel_values_shape}")
            
            # Get IDs
            id_key = "image_ids" if "image_ids" in batch else "image_id"
            if id_key in batch:
                logger.info(f"ID key: {id_key}, Type: {type(batch[id_key])}")
                if isinstance(batch[id_key], torch.Tensor):
                    ids = batch[id_key].cpu().numpy().tolist()
                else:
                    ids = batch[id_key] if isinstance(batch[id_key], list) else [batch[id_key]]
                image_ids.extend(ids)
            
            # Time the model inference
            model_start = time.perf_counter()
            
            # Fix batch dimension before passing to model
            pixel_values = batch["pixel_values"].to(device)
            
            # If 3D (single image), add batch dimension
            if pixel_values.ndim == 3:
                logger.info("Adding missing batch dimension to single image")
                pixel_values = pixel_values.unsqueeze(0)  # [1, 3, 224, 224]
            
            features = model.encode_image(pixel_values)
            if device == "cuda":
                torch.cuda.synchronize()
            model_time = time.perf_counter() - model_start
            logger.info(f"Model inference time: {model_time:.4f}s")
            
            # Time the post-processing
            post_start = time.perf_counter()
            features = features.to(torch.float32)
            features = torch.nn.functional.normalize(features, p=2, dim=-1)
            image_embeddings.append(features.detach().cpu().numpy())
            post_time = time.perf_counter() - post_start
            logger.info(f"Post-processing time: {post_time:.4f}s")
            
            # Overall batch time
            batch_time = time.perf_counter() - batch_start
            logger.info(f"Total batch time: {batch_time:.4f}s")
            logger.info(f"Samples: {batch['pixel_values'].shape[0]}")
            logger.info(f"Processing rate: {batch['pixel_values'].shape[0] / batch_time:.2f} samples/sec")
            
            batch_count += 1
            if batch_count >= batch_limit:
                break
        
        logger.info(f"Processed {batch_count} batches")
        logger.info(f"Collected {len(image_ids)} image IDs")
        
        # Clean up
        del image_embeddings
        del model
        del processor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Error in quick test: {e}", exc_info=True)

if __name__ == "__main__":
    start_time = time.perf_counter()
    asyncio.run(quick_test())
    logger.info(f"Total test time: {time.perf_counter() - start_time:.2f}s") 