#!/usr/bin/env python3
"""
SigLIP model wrapper with dynamic batch sizing support.
"""

import os
import time
import torch
import logging
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SigLIPModelWrapper:
    """
    Wrapper for SigLIP model with dynamic batch sizing and memory management.
    """
    def __init__(
        self,
        model_name: str = "ViT-B-16-SigLIP",
        pretrained: str = "webli",
        device: str = None,
        initial_batch_size: int = 4,
        max_batch_size: int = 32,
        cache_embeddings: bool = True,
    ):
        """
        Initialize the SigLIP model.
        
        Args:
            model_name: SigLIP model name for open_clip (e.g., "ViT-B-16-SigLIP")
            pretrained: Pretrained weights name (e.g., "webli")
            device: Device to load model on ('cpu', 'cuda', 'mps')
            initial_batch_size: Initial batch size for processing
            max_batch_size: Maximum batch size for processing
            cache_embeddings: Whether to cache image embeddings
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_embeddings = cache_embeddings
        
        # Dynamic batch sizing parameters
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.current_batch_size = initial_batch_size
        
        # Embedding cache
        self.embedding_cache = {}
        
        # Load model and processor
        self._load_model()
        logger.info(f"SigLIP model loaded: {self.model_name} with {self.pretrained} weights on {self.device}")
        
    def _load_model(self):
        """Load the SigLIP model and its components."""
        try:
            # Import open_clip
            import open_clip
            
            logger.info(f"Loading SigLIP model using open_clip: {self.model_name} with {self.pretrained} weights")
            
            # Create model and preprocessing transforms
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name, 
                pretrained=self.pretrained
            )
            
            # Create tokenizer
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            
            # Move model to specified device
            self.model = self.model.to(self.device)
            
            # Get embedding dimension
            self.embedding_dim = self.model.embed_dim
            
            # Set model to evaluation mode
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Error loading SigLIP model: {e}")
            raise
    
    def adjust_batch_size(self, processing_time: float, batch_shape: Tuple):
        """
        Adjust batch size based on processing time and memory usage.
        
        Args:
            processing_time: Time taken to process the batch
            batch_shape: Shape of the batch that was processed
            
        Returns:
            New batch size
        """
        # Start with current batch size
        new_batch_size = self.current_batch_size
        
        # Check if we're on CUDA
        if self.device == "cuda" and torch.cuda.is_available():
            # Get current GPU memory usage
            current_memory = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            
            # If memory usage is high (>80%), reduce batch size
            if current_memory > 0.8 and new_batch_size > 1:
                new_batch_size = max(1, new_batch_size - 1)
                logger.info(f"Reducing batch size to {new_batch_size} due to high memory usage ({current_memory:.1%})")
            # If memory usage is low (<50%) and processing time is reasonable (<0.5s per item), increase batch size
            elif current_memory < 0.5 and processing_time / self.current_batch_size < 0.5:
                if new_batch_size < self.max_batch_size:
                    new_batch_size = min(self.max_batch_size, new_batch_size + 1)
                    logger.info(f"Increasing batch size to {new_batch_size} (memory usage: {current_memory:.1%})")
        
        # Update current batch size
        self.current_batch_size = new_batch_size
        return new_batch_size
    
    def get_batch_size(self):
        """Get the current optimal batch size"""
        return self.current_batch_size
    
    @torch.no_grad()
    def encode_images(self, images: List[Union[str, Image.Image]]) -> Dict:
        """
        Encode images using SigLIP's vision encoder.
        
        Args:
            images: List of images to encode (PIL Images or paths)
            
        Returns:
            Dictionary with image embeddings
        """
        # Convert any string paths to PIL Images
        pil_images = []
        cached_indices = []
        uncached_indices = []
        cached_embeddings = []
        
        for i, img in enumerate(images):
            if isinstance(img, str):
                # Check cache first if we're using a path
                if self.cache_embeddings and img in self.embedding_cache:
                    cached_indices.append(i)
                    cached_embeddings.append(self.embedding_cache[img])
                    continue
                
                # Load image if it's a file path
                try:
                    img = Image.open(img).convert('RGB')
                except Exception as e:
                    logger.error(f"Error loading image {img}: {e}")
                    # Create a black image as placeholder
                    img = Image.new('RGB', (224, 224), color=0)
            
            # Add to list of images to process
            pil_images.append(img)
            uncached_indices.append(i)
        
        # Return early if all images were cached
        if not pil_images:
            return {"image_embeddings": torch.stack(cached_embeddings)}
        
        # Measure processing time
        start_time = time.time()
        
        # Preprocess images
        preprocessed_images = torch.stack([self.preprocess(img) for img in pil_images]).to(self.device)
        
        # Get image embeddings
        with torch.no_grad():
            image_embeddings = self.model.encode_image(preprocessed_images)
            # Normalize the embeddings
            image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=1)
        
        # Move results back to CPU
        image_embeddings = image_embeddings.cpu()
        
        # Update cache
        if self.cache_embeddings:
            for i, idx in enumerate(uncached_indices):
                if isinstance(images[idx], str):
                    self.embedding_cache[images[idx]] = image_embeddings[i]
        
        # Record processing time and adjust batch size
        processing_time = time.time() - start_time
        self.adjust_batch_size(
            processing_time=processing_time,
            batch_shape=preprocessed_images.shape
        )
        
        # If we had cached embeddings, integrate them with new embeddings
        if cached_embeddings:
            # Create list of all embeddings in correct order
            all_embeddings = [None] * len(images)
            for i, embedding in zip(cached_indices, cached_embeddings):
                all_embeddings[i] = embedding
            for i, embedding in zip(uncached_indices, image_embeddings):
                all_embeddings[i] = embedding
            
            image_embeddings = torch.stack(all_embeddings)
        
        return {
            "image_embeddings": image_embeddings,
            "processing_time": processing_time,
            "batch_size": len(pil_images)
        }
    
    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> Dict:
        """
        Encode text using SigLIP's text encoder.
        
        Args:
            texts: List of text inputs
            
        Returns:
            Dictionary with text embeddings
        """
        # Measure processing time
        start_time = time.time()
        
        # Tokenize texts
        tokenized_texts = self.tokenizer(texts).to(self.device)
        
        # Get text embeddings
        with torch.no_grad():
            text_embeddings = self.model.encode_text(tokenized_texts)
            # Normalize the embeddings
            text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=1)
        
        # Move results back to CPU
        text_embeddings = text_embeddings.cpu()
        
        # Record processing time and adjust batch size
        processing_time = time.time() - start_time
        self.adjust_batch_size(
            processing_time=processing_time,
            batch_shape=tokenized_texts.shape
        )
        
        return {
            "text_embeddings": text_embeddings,
            "processing_time": processing_time,
            "batch_size": len(texts)
        }
    
    def compute_similarity(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between image and text embeddings.
        
        Args:
            image_embeddings: Image embeddings (batch_size, embedding_dim)
            text_embeddings: Text embeddings (batch_size, embedding_dim)
            
        Returns:
            Cosine similarity matrix (batch_size, batch_size)
        """
        # Ensure embeddings are on CPU and normalized
        if image_embeddings.device != torch.device("cpu"):
            image_embeddings = image_embeddings.cpu()
        if text_embeddings.device != torch.device("cpu"):
            text_embeddings = text_embeddings.cpu()
            
        # Normalize embeddings if not already normalized
        image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=1)
        text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.matmul(image_embeddings, text_embeddings.T)
        
        return similarity

# Helper function to load SigLIP model
def load_siglip_model(
    model_name: str = "ViT-B-16-SigLIP",
    pretrained: str = "webli",
    device: str = None,
    initial_batch_size: int = 4,
    max_batch_size: int = 32,
    cache_embeddings: bool = True,
) -> SigLIPModelWrapper:
    """
    Load SigLIP model with specified parameters.
    
    Args:
        model_name: SigLIP model name for open_clip
        pretrained: Pretrained weights name
        device: Device to load model on
        initial_batch_size: Initial batch size for processing
        max_batch_size: Maximum batch size for processing
        cache_embeddings: Whether to cache image embeddings
        
    Returns:
        Initialized SigLIPModelWrapper
    """
    return SigLIPModelWrapper(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
        initial_batch_size=initial_batch_size,
        max_batch_size=max_batch_size,
        cache_embeddings=cache_embeddings,
    ) 