"""
Inference Pipeline

This module implements the core inference pipeline for CLIP embedding generation.
"""
import asyncio
import io
import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from batching.dynamic_batcher import DynamicBatcher
from models.registry import ModelRegistry
from optimization.profiler import profile_inference

logger = logging.getLogger(__name__)


class InferencePipeline:
    """CLIP Inference Pipeline for generating embeddings."""
    
    def __init__(self, model, config):
        """Initialize the inference pipeline."""
        self.model = model
        self.config = config
        self.registry = ModelRegistry(config)
        
        # Initialize dynamic batcher
        self.batcher = DynamicBatcher(
            batch_size=config["inference"]["batch_size"],
            max_batch_size=config["inference"]["max_batch_size"],
            min_batch_size=config["inference"]["min_batch_size"],
            max_wait_ms=config["inference"]["max_wait_ms"],
        )
        
        # Set device
        self.device = torch.device(config["model"]["device"])
        
        # Initialize cache if enabled
        self.cache = {}  # Simple in-memory cache for now
        
        logger.info(f"Inference pipeline initialized with model: {model.model_name}")
    
    async def encode_text(
        self, 
        texts: List[str],
        model_name: Optional[str] = None,
        normalize: bool = True
    ) -> Dict:
        """
        Generate embeddings for text inputs.
        
        Args:
            texts: List of text inputs
            model_name: Optional model name to use (default: use configured model)
            normalize: Whether to normalize embeddings to unit length
            
        Returns:
            Dictionary with embeddings and metadata
        """
        # Handle model selection
        model = self.model if model_name is None else self.registry.get_model(model_name)
        
        # Check cache for each text
        cached_results = {}
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = f"text:{text}:{model.model_name}:{normalize}"
            if cache_key in self.cache:
                cached_results[i] = self.cache[cache_key]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Process uncached texts if any
        if uncached_texts:
            # Add to batch queue and wait for results
            start_time = time.time()
            
            # Prepare batch request
            batch_request = {
                "texts": uncached_texts,
                "model": model,
                "normalize": normalize,
            }
            
            # Process through batcher
            batch_results = await self.batcher.process(
                batch_request,
                self._process_text_batch
            )
            
            # Cache results
            for i, text in zip(uncached_indices, uncached_texts):
                embedding = batch_results["embeddings"][i - uncached_indices[0]]
                cache_key = f"text:{text}:{model.model_name}:{normalize}"
                self.cache[cache_key] = embedding
                cached_results[i] = embedding
            
            elapsed = (time.time() - start_time) * 1000
            logger.debug(f"Text embedding generation took {elapsed:.2f}ms for {len(uncached_texts)} texts")
        
        # Combine cached and new results
        all_embeddings = []
        for i in range(len(texts)):
            all_embeddings.append(cached_results[i])
        
        return {
            "embeddings": np.array(all_embeddings),
            "model": model.model_name,
            "dimensions": model.embedding_dim,
        }
    
    async def encode_image_from_urls(
        self,
        image_urls: List[str],
        model_name: Optional[str] = None,
        normalize: bool = True,
        timeout: float = 30.0
    ) -> Dict:
        """
        Generate embeddings for images from URLs.
        
        Args:
            image_urls: List of image URLs
            model_name: Optional model name to use
            normalize: Whether to normalize embeddings
            timeout: Timeout in seconds for URL fetching
            
        Returns:
            Dictionary with embeddings and metadata
        """
        # Handle model selection
        model = self.model if model_name is None else self.registry.get_model(model_name)
        
        # Check cache and prepare uncached requests
        cached_results = {}
        uncached_urls = []
        uncached_indices = []
        
        for i, url in enumerate(image_urls):
            cache_key = f"image_url:{url}:{model.model_name}:{normalize}"
            if cache_key in self.cache:
                cached_results[i] = self.cache[cache_key]
            else:
                uncached_urls.append(url)
                uncached_indices.append(i)
        
        # Process uncached URLs if any
        if uncached_urls:
            start_time = time.time()
            
            # Download images asynchronously
            images = await self._fetch_images(uncached_urls, timeout)
            
            # Prepare batch request
            batch_request = {
                "images": images,
                "model": model,
                "normalize": normalize,
            }
            
            # Process through batcher
            batch_results = await self.batcher.process(
                batch_request,
                self._process_image_batch
            )
            
            # Cache results
            for i, url in zip(uncached_indices, uncached_urls):
                embedding = batch_results["embeddings"][i - uncached_indices[0]]
                cache_key = f"image_url:{url}:{model.model_name}:{normalize}"
                self.cache[cache_key] = embedding
                cached_results[i] = embedding
            
            elapsed = (time.time() - start_time) * 1000
            logger.debug(f"Image embedding generation took {elapsed:.2f}ms for {len(uncached_urls)} images")
        
        # Combine cached and new results
        all_embeddings = []
        for i in range(len(image_urls)):
            all_embeddings.append(cached_results[i])
        
        return {
            "embeddings": np.array(all_embeddings),
            "model": model.model_name,
            "dimensions": model.embedding_dim,
        }

    async def encode_image_from_bytes(
        self,
        image_bytes: List[bytes],
        model_name: Optional[str] = None,
        normalize: bool = True
    ) -> Dict:
        """
        Generate embeddings for images from bytes.
        
        Args:
            image_bytes: List of image bytes
            model_name: Optional model name to use
            normalize: Whether to normalize embeddings
            
        Returns:
            Dictionary with embeddings and metadata
        """
        # Handle model selection
        model = self.model if model_name is None else self.registry.get_model(model_name)
        
        # Convert bytes to PIL Images
        images = [Image.open(io.BytesIO(img_bytes)) for img_bytes in image_bytes]
        
        # Prepare batch request
        batch_request = {
            "images": images,
            "model": model,
            "normalize": normalize,
        }
        
        # Process through batcher
        start_time = time.time()
        results = await self.batcher.process(
            batch_request,
            self._process_image_batch
        )
        
        elapsed = (time.time() - start_time) * 1000
        logger.debug(f"Image embedding generation took {elapsed:.2f}ms for {len(images)} images")
        
        return results

    @profile_inference
    async def _process_text_batch(self, batch: Dict) -> Dict:
        """Process a batch of text inputs."""
        texts = batch["texts"]
        model = batch["model"]
        normalize = batch["normalize"]
        
        # Move model to correct device if needed
        model.to(self.device)
        
        with torch.no_grad():
            # Generate embeddings
            embeddings = model.encode_text(texts)
            
            # Normalize if requested
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            
            # Move to CPU and convert to numpy
            embeddings = embeddings.cpu().numpy()
        
        return {
            "embeddings": embeddings,
            "model": model.model_name,
            "dimensions": model.embedding_dim,
        }

    @profile_inference
    async def _process_image_batch(self, batch: Dict) -> Dict:
        """Process a batch of image inputs."""
        images = batch["images"]
        model = batch["model"]
        normalize = batch["normalize"]
        
        # Move model to correct device if needed
        model.to(self.device)
        
        with torch.no_grad():
            # Generate embeddings
            embeddings = model.encode_image(images)
            
            # Normalize if requested
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            
            # Move to CPU and convert to numpy
            embeddings = embeddings.cpu().numpy()
        
        return {
            "embeddings": embeddings,
            "model": model.model_name,
            "dimensions": model.embedding_dim,
        }

    async def _fetch_images(
        self,
        urls: List[str],
        timeout: float
    ) -> List[Image.Image]:
        """
        Fetch images from URLs asynchronously.
        
        Args:
            urls: List of image URLs
            timeout: Timeout in seconds
            
        Returns:
            List of PIL Images
        """
        import aiohttp
        from aiohttp import ClientTimeout
        
        async def fetch_single_image(session: aiohttp.ClientSession, url: str) -> Image.Image:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.read()
                        return Image.open(io.BytesIO(data))
                    else:
                        raise ValueError(f"Failed to fetch image from {url}: {response.status}")
            except Exception as e:
                logger.error(f"Error fetching image from {url}: {str(e)}")
                raise
        
        timeout_config = ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_config) as session:
            tasks = [fetch_single_image(session, url) for url in urls]
            return await asyncio.gather(*tasks)

    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache.clear()
        logger.info("Inference pipeline cache cleared")

    def get_cache_size(self) -> int:
        """Get the current size of the embedding cache."""
        return len(self.cache)

    def warmup(self, num_warmup_runs: int = 3):
        """
        Perform warmup runs to initialize model and cache GPU memory.
        
        Args:
            num_warmup_runs: Number of warmup runs to perform
        """
        logger.info(f"Performing {num_warmup_runs} warmup runs...")
        
        # Warmup text encoding
        dummy_texts = ["warmup text"] * 2
        for _ in range(num_warmup_runs):
            asyncio.run(self.encode_text(dummy_texts))
        
        # Warmup image encoding
        dummy_image = Image.new('RGB', (224, 224))
        dummy_images = [dummy_image] * 2
        for _ in range(num_warmup_runs):
            asyncio.run(self.encode_image_from_bytes([dummy_image.tobytes() for _ in range(2)]))
        
        logger.info("Warmup complete")