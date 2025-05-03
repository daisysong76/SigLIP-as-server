#!/usr/bin/env python3
"""
LLaVA model wrapper with dynamic batch sizing support.
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

class LLaVAModelWrapper:
    """
    Wrapper for LLaVA model with dynamic batch sizing and memory management.
    """
    def __init__(
        self,
        model_path: str = "liuhaotian/llava-v1.5-7b",
        device: str = None,
        load_8bit: bool = False,
        load_4bit: bool = False,
        image_processor = None,
        initial_batch_size: int = 1,
        max_batch_size: int = 4,
        cache_embeddings: bool = True,
    ):
        """
        Initialize the LLaVA model.
        
        Args:
            model_path: Path to LLaVA model
            device: Device to load model on ('cpu', 'cuda', 'mps')
            load_8bit: Whether to load model in 8-bit precision
            load_4bit: Whether to load model in 4-bit precision
            image_processor: Optional pre-initialized image processor
            initial_batch_size: Initial batch size for processing
            max_batch_size: Maximum batch size for processing
            cache_embeddings: Whether to cache image embeddings
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_8bit = load_8bit
        self.load_4bit = load_4bit
        self.cache_embeddings = cache_embeddings
        
        # Dynamic batch sizing parameters
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.current_batch_size = initial_batch_size
        
        # Embedding cache
        self.embedding_cache = {}
        
        # Load model and processor
        self._load_model(image_processor)
        logger.info(f"LLaVA model loaded from {model_path} on {self.device}")
        
    def _load_model(self, image_processor=None):
        """Load the LLaVA model and its components."""
        try:
            # Import here to avoid dependency issues if LLaVA is not installed
            from transformers import AutoProcessor, AutoModelForCausalLM
            import torch
            
            logger.info(f"Loading LLaVA model: {self.model_path}")
            
            # Load with appropriate quantization settings
            if self.load_8bit:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                    device_map="auto"
                )
            elif self.load_4bit:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    load_in_4bit=True,
                    device_map="auto"
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            
            # Load processor
            if image_processor is None:
                self.processor = AutoProcessor.from_pretrained(self.model_path)
            else:
                self.processor = image_processor
            
            # Get model dimensions and capabilities
            self.embedding_dim = self.model.config.hidden_size
            self.context_length = getattr(self.model.config, "max_position_embeddings", 2048)
            
            # Set model to evaluation mode
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Error loading LLaVA model: {e}")
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
        Encode images using LLaVA's vision encoder.
        
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
        
        # Process images through model's vision encoder
        inputs = self.processor(
            images=pil_images, 
            return_tensors="pt"
        ).to(self.device)
        
        # Get image embeddings - we don't generate text yet
        with torch.no_grad():
            image_embeddings = self.model.get_image_features(**inputs)
            
            # Some LLaVA models may not have direct access to image embeddings
            # In that case, we need to modify this approach
            if image_embeddings is None:
                logger.warning("Direct image embeddings not available for this model. Using processed input instead.")
                image_embeddings = inputs.pixel_values
        
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
            batch_shape=inputs.pixel_values.shape
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
    def generate_response(
        self, 
        images: List[Union[str, Image.Image]], 
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_beams: int = 1
    ) -> Dict:
        """
        Generate LLaVA responses for a batch of images and prompts.
        
        Args:
            images: List of images (PIL Images or paths)
            prompts: List of text prompts
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            num_beams: Number of beams for beam search
            
        Returns:
            Dictionary with generated responses
        """
        assert len(images) == len(prompts), "Number of images must match number of prompts"
        
        # Measure processing time
        start_time = time.time()
        
        # Process inputs
        inputs = self.processor(
            images=images,
            text=prompts,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "num_beams": num_beams,
            "use_cache": True,
            "do_sample": temperature > 0,
        }
        
        # Some models might need different generation configurations
        if hasattr(self.model, "generate"):
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    **generation_args
                )
        else:
            logger.error("Model does not have generate method")
            return {"error": "Model does not support text generation"}
        
        # Decode outputs
        responses = self.processor.batch_decode(
            output_ids, 
            skip_special_tokens=True
        )
        
        # Record processing time and adjust batch size
        processing_time = time.time() - start_time
        self.adjust_batch_size(
            processing_time=processing_time,
            batch_shape=inputs.pixel_values.shape if hasattr(inputs, "pixel_values") else (len(images), 1)
        )
        
        return {
            "responses": responses,
            "processing_time": processing_time,
            "batch_size": len(images)
        }

# Helper function to load LLaVA model
def load_llava_model(
    model_path: str = "liuhaotian/llava-v1.5-7b",
    device: str = None,
    load_8bit: bool = False,
    load_4bit: bool = False,
    initial_batch_size: int = 1,
    max_batch_size: int = 4,
    cache_embeddings: bool = True,
) -> LLaVAModelWrapper:
    """
    Load LLaVA model with specified parameters.
    
    Args:
        model_path: Path to LLaVA model
        device: Device to load model on
        load_8bit: Whether to load model in 8-bit precision
        load_4bit: Whether to load model in 4-bit precision
        initial_batch_size: Initial batch size for processing
        max_batch_size: Maximum batch size for processing
        cache_embeddings: Whether to cache image embeddings
        
    Returns:
        Initialized LLaVAModelWrapper
    """
    return LLaVAModelWrapper(
        model_path=model_path,
        device=device,
        load_8bit=load_8bit,
        load_4bit=load_4bit,
        initial_batch_size=initial_batch_size,
        max_batch_size=max_batch_size,
        cache_embeddings=cache_embeddings,
    ) 