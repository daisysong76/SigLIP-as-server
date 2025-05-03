#!/usr/bin/env python3
"""
Advanced SigLIP model implementation with industry best practices.

This module provides a production-ready SigLIP implementation with:
- Automatic mixed precision
- Efficient batching
- Memory optimization
- Caching infrastructure
- Lazy loading
- Quantization options
- Dynamic resource adaptation
"""

import os
import time
import torch
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import json
from functools import lru_cache
from contextlib import nullcontext

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
EMBEDDING_DIMENSION_MAP = {
    "ViT-B-16-SigLIP": 768,
    "ViT-L-14-SigLIP": 1024,
    "ViT-H-14-SigLIP": 1280,
    "ViT-g-14-SigLIP": 1536,
}

@dataclass
class SigLIPConfig:
    """Configuration settings for SigLIP model."""
    model_name: str = "ViT-B-16-SigLIP"
    pretrained: str = "webli"
    device: Optional[str] = None
    precision: str = "fp16"  # "fp32", "fp16", "bf16", "int8"
    initial_batch_size: int = 4
    max_batch_size: int = 32
    cache_embeddings: bool = True
    cache_dir: Optional[str] = None
    jit_mode: bool = False
    dynamic_resource_allocation: bool = True
    prefetch: bool = True
    embedding_dimension: Optional[int] = None
    enable_checkpointing: bool = False
    distributed: bool = False
    gradient_checkpointing: bool = False  # Memory optimization for larger models


class AdvancedSigLIPModel:
    """
    Advanced SigLIP model with industry best practices for production deployment.
    """
    def __init__(self, config: SigLIPConfig = None):
        """
        Initialize the SigLIP model with advanced configuration.
        
        Args:
            config: SigLIPConfig object with model settings
        """
        self.config = config or SigLIPConfig()
        
        # Resolve device
        self.device = self._resolve_device()
        
        # Determine precision
        self.precision, self.amp_dtype = self._resolve_precision()
        
        # Set up caching
        self._setup_cache()
        
        # Embedding dimension
        self.embedding_dim = self.config.embedding_dimension or EMBEDDING_DIMENSION_MAP.get(
            self.config.model_name, 768)
        
        # Dynamic resource allocation settings
        self.current_batch_size = self.config.initial_batch_size
        self.last_memory_check = 0
        self.memory_check_interval = 10  # Check memory every 10 batches
        
        # Lazy loading - only load when needed
        self._model = None
        self._tokenizer = None
        self._preprocessor = None
        self._jit_model = None
        self._is_loaded = False
        
    def _resolve_device(self) -> str:
        """Resolve the device to use for model inference."""
        if self.config.device:
            return self.config.device
            
        # Auto-detect device
        if torch.cuda.is_available():
            # Get device with most available memory
            if self.config.dynamic_resource_allocation and torch.cuda.device_count() > 1:
                max_free_mem = 0
                best_device_idx = 0
                
                for i in range(torch.cuda.device_count()):
                    free_mem = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                    if free_mem > max_free_mem:
                        max_free_mem = free_mem
                        best_device_idx = i
                        
                return f"cuda:{best_device_idx}"
            else:
                return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon GPU
        else:
            return "cpu"
            
    def _resolve_precision(self) -> Tuple[str, torch.dtype]:
        """Resolve precision and corresponding dtype."""
        precision = self.config.precision.lower()
        device = self.device
        
        # CPU only supports fp32
        if device == "cpu" and precision != "fp32":
            logger.info("CPU only supports fp32 precision, ignoring requested precision")
            return "fp32", torch.float32
            
        # Handle different precision options
        if precision == "fp16" and device.startswith("cuda"):
            return "fp16", torch.float16
        elif precision == "bf16" and device.startswith("cuda") and torch.cuda.is_bf16_supported():
            return "bf16", torch.bfloat16
        elif precision == "int8":
            # Will apply quantization later
            return "int8", torch.qint8
        else:
            # Default to fp32
            return "fp32", torch.float32
            
    def _setup_cache(self) -> None:
        """Set up embedding cache infrastructure."""
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        if self.config.cache_embeddings and self.config.cache_dir:
            os.makedirs(self.config.cache_dir, exist_ok=True)
            
            # Load existing cache if available
            cache_file = os.path.join(self.config.cache_dir, f"siglip_{self.config.model_name}_cache.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        cache_info = json.load(f)
                        logger.info(f"Loaded cache info: {len(cache_info)} embeddings indexed")
                        self.cache_index = cache_info
                except Exception as e:
                    logger.warning(f"Failed to load cache index: {e}")
                    self.cache_index = {}
            else:
                self.cache_index = {}
    
    def load(self) -> None:
        """
        Load the SigLIP model and its components.
        
        This is called automatically on first use, or can be called explicitly.
        """
        if self._is_loaded:
            return
            
        logger.info(f"Loading SigLIP model: {self.config.model_name} with {self.config.pretrained} weights")
        
        try:
            # Import open_clip
            import open_clip
            
            # Create model and preprocessing transforms
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.config.model_name,
                pretrained=self.config.pretrained
            )
            
            # Create tokenizer
            tokenizer = open_clip.get_tokenizer(self.config.model_name)
            
            # Apply precision transformations
            if self.precision == "fp16":
                model = model.half()
            elif self.precision == "int8":
                # Quantization for inference
                from torch.quantization import quantize_dynamic
                model = quantize_dynamic(
                    model, 
                    {torch.nn.Linear}, 
                    dtype=torch.qint8
                )
                logger.info("Applied INT8 quantization")
            
            # Move model to device
            model = model.to(self.device)
            
            # Apply JIT compilation if configured
            if self.config.jit_mode:
                try:
                    # Script the model
                    dummy_image_input = torch.randn(1, 3, 224, 224, device=self.device)
                    dummy_text_input = torch.zeros(1, 77, dtype=torch.long, device=self.device)
                    
                    # Trace image encoder
                    image_encoder = torch.jit.trace(
                        model.encode_image, 
                        (dummy_image_input,)
                    )
                    
                    # Trace text encoder
                    text_encoder = torch.jit.trace(
                        model.encode_text,
                        (dummy_text_input,)
                    )
                    
                    # Replace model components with JIT compiled versions
                    model.encode_image = image_encoder
                    model.encode_text = text_encoder
                    
                    logger.info("Applied JIT compilation to model")
                except Exception as e:
                    logger.warning(f"JIT compilation failed: {e}. Using eager mode.")
            
            # Enable gradient checkpointing for memory efficiency if configured
            if self.config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing")
            
            # Save model components
            self._model = model
            self._tokenizer = tokenizer
            self._preprocessor = preprocess
            
            # Set model to evaluation mode
            self._model.eval()
            self._is_loaded = True
            
            logger.info(f"SigLIP model loaded successfully on {self.device} with {self.precision} precision")
            
        except Exception as e:
            logger.error(f"Error loading SigLIP model: {e}")
            raise
    
    def adjust_batch_size(self, processing_time: float, batch_size: int, shape: tuple) -> int:
        """
        Dynamically adjust batch size based on processing time and memory.
        
        Args:
            processing_time: Time taken to process the batch
            batch_size: Current batch size
            shape: Shape of the processed batch
            
        Returns:
            New optimal batch size
        """
        if not self.config.dynamic_resource_allocation:
            return self.current_batch_size
            
        # Measure time per sample
        time_per_sample = processing_time / batch_size
        
        # Check if we should adjust batch size
        current_time = time.time()
        if current_time - self.last_memory_check > self.memory_check_interval:
            self.last_memory_check = current_time
            
            # Check memory usage on appropriate device
            if self.device.startswith("cuda"):
                # Get current GPU memory usage
                current_memory = torch.cuda.memory_allocated(self.device) / \
                                torch.cuda.get_device_properties(0).total_memory
                
                # Adjust batch size based on memory usage
                if current_memory > 0.85 and self.current_batch_size > self.config.initial_batch_size:
                    # Reduce batch size if memory usage is high
                    new_batch_size = max(self.config.initial_batch_size, self.current_batch_size - 4)
                    logger.info(f"Reducing batch size to {new_batch_size} due to high memory usage ({current_memory:.1%})")
                    self.current_batch_size = new_batch_size
                elif current_memory < 0.5 and time_per_sample < 0.05:
                    # Increase batch size if memory usage is low and processing is fast
                    new_batch_size = min(self.current_batch_size + 4, self.config.max_batch_size)
                    if new_batch_size > self.current_batch_size:
                        logger.info(f"Increasing batch size to {new_batch_size} (memory: {current_memory:.1%})")
                        self.current_batch_size = new_batch_size
            else:
                # On CPU, adjust based on time per sample only
                if time_per_sample < 0.01 and self.current_batch_size < self.config.max_batch_size:
                    self.current_batch_size = min(self.current_batch_size + 2, self.config.max_batch_size)
                elif time_per_sample > 0.1 and self.current_batch_size > self.config.initial_batch_size:
                    self.current_batch_size = max(self.config.initial_batch_size, self.current_batch_size - 2)
                    
        return self.current_batch_size
    
    def _ensure_loaded(self) -> None:
        """Ensure model is loaded before use."""
        if not self._is_loaded:
            self.load()
    
    @torch.no_grad()
    def encode_images(self, images: List[Union[str, Image.Image]], batch_size: Optional[int] = None) -> Dict:
        """
        Encode images with SigLIP model, with optimized batching and caching.
        
        Args:
            images: List of images (PIL Images or file paths)
            batch_size: Batch size for processing (overrides current dynamic batch size)
            
        Returns:
            Dictionary with image embeddings and metadata
        """
        self._ensure_loaded()
        
        batch_size = batch_size or self.current_batch_size
        processed_images = []
        image_sources = []
        cached_indices = []
        uncached_indices = []
        cached_embeddings = []
        
        # Set up mixed precision context
        amp_context = torch.cuda.amp.autocast(dtype=self.amp_dtype) if \
                      self.precision in ["fp16", "bf16"] and self.device != "cpu" else \
                      nullcontext()
        
        # Check cache first
        for i, img in enumerate(images):
            cache_key = str(img) if isinstance(img, str) else None
            
            if cache_key and self.config.cache_embeddings and cache_key in self.embedding_cache:
                cached_indices.append(i)
                cached_embeddings.append(self.embedding_cache[cache_key])
                self.cache_hits += 1
                continue
            
            # Load image if needed
            if isinstance(img, str):
                try:
                    img = Image.open(img).convert('RGB')
                except Exception as e:
                    logger.warning(f"Error loading image {img}: {e}. Using placeholder.")
                    img = Image.new('RGB', (224, 224), color=0)
            
            # Add to processing list
            processed_images.append(img)
            image_sources.append(cache_key)
            uncached_indices.append(i)
            self.cache_misses += 1
        
        # Return early if all images were cached
        if not processed_images:
            return {
                "image_embeddings": torch.stack(cached_embeddings),
                "cache_hit_rate": 1.0,
                "processing_time": 0.0
            }
        
        # Process in batches
        start_time = time.time()
        all_embeddings = []
        
        # Create batches
        for batch_idx in range(0, len(processed_images), batch_size):
            batch_images = processed_images[batch_idx:batch_idx + batch_size]
            batch_sources = image_sources[batch_idx:batch_idx + batch_size]
            
            # Preprocess
            try:
                preprocessed = torch.stack([self._preprocessor(img) for img in batch_images]).to(self.device)
                
                # Clear memory between batch preprocessing and inference
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                
                # Forward pass with mixed precision
                with amp_context:
                    embeddings = self._model.encode_image(preprocessed)
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                # Add to results
                all_embeddings.append(embeddings.cpu())
                
                # Update cache
                if self.config.cache_embeddings:
                    for idx, src in enumerate(batch_sources):
                        if src:  # Only cache if we have a valid source (filepath)
                            self.embedding_cache[src] = embeddings[idx].cpu()
            
            except Exception as e:
                logger.error(f"Error processing image batch: {e}")
                # Create empty embeddings as fallback
                empty_shape = (len(batch_images), self.embedding_dim)
                all_embeddings.append(torch.zeros(empty_shape))
        
        # Measure processing time
        processing_time = time.time() - start_time
        
        # Adjust batch size for future calls
        self.adjust_batch_size(
            processing_time=processing_time,
            batch_size=len(processed_images),
            shape=(len(processed_images), 3, 224, 224)
        )
        
        # Combine all embeddings
        new_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Combine with cached embeddings if needed
        if cached_embeddings:
            # Create final embeddings tensor in correct order
            all_embeddings = torch.zeros((len(images), self.embedding_dim))
            
            # Place cached embeddings
            for idx, embedding in zip(cached_indices, cached_embeddings):
                all_embeddings[idx] = embedding
                
            # Place new embeddings
            for idx, embedding in zip(uncached_indices, new_embeddings):
                all_embeddings[idx] = embedding
        else:
            all_embeddings = new_embeddings
        
        # Calculate cache statistics
        cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if self.cache_hits > 0 else 0
        
        return {
            "image_embeddings": all_embeddings,
            "processing_time": processing_time,
            "batch_size": self.current_batch_size,
            "cache_hit_rate": cache_hit_rate
        }
    
    @torch.no_grad()
    def encode_text(self, texts: List[str], batch_size: Optional[int] = None) -> Dict:
        """
        Encode text with SigLIP model, with optimized batching.
        
        Args:
            texts: List of text inputs
            batch_size: Batch size for processing (overrides current batch size)
            
        Returns:
            Dictionary with text embeddings and metadata
        """
        self._ensure_loaded()
        
        batch_size = batch_size or self.current_batch_size
        start_time = time.time()
        all_embeddings = []
        
        # Set up mixed precision context
        amp_context = torch.cuda.amp.autocast(dtype=self.amp_dtype) if \
                      self.precision in ["fp16", "bf16"] and self.device != "cpu" else \
                      nullcontext()
        
        # Process in batches
        for batch_idx in range(0, len(texts), batch_size):
            batch_texts = texts[batch_idx:batch_idx + batch_size]
            
            try:
                # Tokenize
                tokenized = self._tokenizer(batch_texts).to(self.device)
                
                # Forward pass with mixed precision
                with amp_context:
                    embeddings = self._model.encode_text(tokenized)
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                # Add to results
                all_embeddings.append(embeddings.cpu())
                
            except Exception as e:
                logger.error(f"Error processing text batch: {e}")
                # Create empty embeddings as fallback
                empty_shape = (len(batch_texts), self.embedding_dim)
                all_embeddings.append(torch.zeros(empty_shape))
        
        # Measure processing time
        processing_time = time.time() - start_time
        
        # Adjust batch size for future calls
        self.adjust_batch_size(
            processing_time=processing_time,
            batch_size=len(texts),
            shape=(len(texts), 77)  # Typical tokenized shape
        )
        
        # Combine all embeddings
        text_embeddings = torch.cat(all_embeddings, dim=0)
        
        return {
            "text_embeddings": text_embeddings,
            "processing_time": processing_time,
            "batch_size": self.current_batch_size
        }
    
    def compute_similarity(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between image and text embeddings efficiently.
        
        Args:
            image_embeddings: Image embeddings (N x embedding_dim)
            text_embeddings: Text embeddings (M x embedding_dim)
            
        Returns:
            Cosine similarity matrix (N x M)
        """
        # Ensure embeddings are on CPU and normalized
        if image_embeddings.device != torch.device("cpu"):
            image_embeddings = image_embeddings.cpu()
        if text_embeddings.device != torch.device("cpu"):
            text_embeddings = text_embeddings.cpu()
        
        # Ensure normalization
        image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=1)
        text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=1)
        
        # Use efficient batch processing for large matrices
        if image_embeddings.shape[0] * text_embeddings.shape[0] > 1e6:
            return self._batched_similarity(image_embeddings, text_embeddings)
        else:
            # Direct computation for smaller matrices
            return torch.matmul(image_embeddings, text_embeddings.T)
    
    def _batched_similarity(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor, 
                           batch_size: int = 1024) -> torch.Tensor:
        """
        Compute similarity in batches to avoid OOM for large matrices.
        
        Args:
            image_embeddings: Image embeddings
            text_embeddings: Text embeddings
            batch_size: Maximum batch size for computation
            
        Returns:
            Similarity matrix
        """
        num_images = image_embeddings.shape[0]
        num_texts = text_embeddings.shape[0]
        similarity = torch.zeros((num_images, num_texts))
        
        for i in range(0, num_images, batch_size):
            img_batch = image_embeddings[i:i+batch_size]
            
            for j in range(0, num_texts, batch_size):
                txt_batch = text_embeddings[j:j+batch_size]
                
                # Compute partial similarity matrix
                similarity[i:i+batch_size, j:j+batch_size] = torch.matmul(img_batch, txt_batch.T)
        
        return similarity
    
    def save_cache(self) -> None:
        """Save embedding cache to disk if configured."""
        if not self.config.cache_embeddings or not self.config.cache_dir:
            return
            
        # Save cache index
        cache_file = os.path.join(self.config.cache_dir, f"siglip_{self.config.model_name}_cache.json")
        
        try:
            cache_info = {}
            for key in self.embedding_cache:
                # We only store metadata in the index file
                cache_info[key] = {
                    "shape": list(self.embedding_cache[key].shape),
                    "created": time.time()
                }
                
            with open(cache_file, 'w') as f:
                json.dump(cache_info, f)
                
            logger.info(f"Saved cache index with {len(cache_info)} entries to {cache_file}")
            
            # Save actual embeddings
            embeddings_file = os.path.join(self.config.cache_dir, f"siglip_{self.config.model_name}_embeddings.pt")
            torch.save(self.embedding_cache, embeddings_file)
            logger.info(f"Saved {len(self.embedding_cache)} embeddings to {embeddings_file}")
            
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Cleared embedding cache")
    
    def get_batch_size(self) -> int:
        """Get the current optimal batch size."""
        return self.current_batch_size
    
    def __del__(self):
        """Cleanup when object is deleted."""
        if hasattr(self, 'config') and self.config.cache_embeddings:
            try:
                self.save_cache()
            except:
                pass

# Factory function to create the advanced model
def create_advanced_siglip_model(
    model_name: str = "ViT-B-16-SigLIP",
    pretrained: str = "webli",
    device: Optional[str] = None,
    precision: str = "fp16",
    initial_batch_size: int = 4,
    max_batch_size: int = 32,
    cache_embeddings: bool = True,
    cache_dir: Optional[str] = None,
    jit_mode: bool = False,
    **kwargs
) -> AdvancedSigLIPModel:
    """
    Create an advanced SigLIP model with industry best practices.
    
    Args:
        model_name: SigLIP model architecture name
        pretrained: Pretrained weights identifier
        device: Device for inference
        precision: Precision for inference
        initial_batch_size: Initial batch size for processing
        max_batch_size: Maximum batch size for processing
        cache_embeddings: Whether to cache embeddings
        cache_dir: Directory to cache embeddings
        jit_mode: Whether to apply JIT compilation
        **kwargs: Additional configuration options
        
    Returns:
        Configured AdvancedSigLIPModel instance
    """
    config = SigLIPConfig(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
        precision=precision,
        initial_batch_size=initial_batch_size,
        max_batch_size=max_batch_size,
        cache_embeddings=cache_embeddings,
        cache_dir=cache_dir,
        jit_mode=jit_mode,
        **kwargs
    )
    
    return AdvancedSigLIPModel(config) 