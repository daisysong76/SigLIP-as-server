"""Redis-based advanced caching for CLIP embeddings with batch operations and metrics."""
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
import time
import json
import hashlib
from redis.asyncio import Redis, from_url
from loguru import logger


class RedisEmbeddingCache:
    def __init__(self, config: Dict[str, Any]):
        """Initialize Redis cache for CLIP embeddings with advanced features.
        
        Args:
            config: Configuration dictionary containing Redis settings and cache options
        """
        # Redis connection setup
        self.redis: Redis = from_url(
            f"redis://{config['redis']['host']}:{config['redis']['port']}",
            password=config["redis"]["password"],
            db=config["redis"]["db"],
            encoding="utf-8",
            decode_responses=False  # We want raw bytes for numpy arrays
        )
        
        # Cache settings
        self.ttl = config["redis"].get("ttl", 86400)  # Default 1 day TTL
        self.namespace = config["redis"].get("namespace", "clip:")
        self.compression = config["redis"].get("compression", False)
        
        # Metrics tracking
        self.hits = 0
        self.misses = 0
        self.total_saved_time = 0
        self._start_time = time.time()
        
        # Pipeline configuration
        self.max_pipeline_size = config["redis"].get("max_pipeline_size", 100)
        
        logger.info(f"Redis CLIP embedding cache initialized with namespace '{self.namespace}'")
        
        if self.compression:
            try:
                import zstd
                self._compress = zstd.compress
                self._decompress = zstd.decompress
                logger.info("Using zstd compression for embeddings")
            except ImportError:
                logger.warning("zstd compression requested but not installed. Running without compression.")
                self.compression = False
    
    def _format_key(self, key: str) -> str:
        """Create namespaced Redis key with hash for long keys."""
        if len(key) > 100:  # Hash long keys
            key = hashlib.md5(key.encode()).hexdigest()
        return f"{self.namespace}{key}"
    
    async def get(self, key: str) -> Optional[np.ndarray]:
        """Retrieve a single embedding from cache.
        
        Args:
            key: The cache key to retrieve
            
        Returns:
            The embedding as numpy array or None if not found
        """
        try:
            start_time = time.time()
            formatted_key = self._format_key(key)
            value = await self.redis.get(formatted_key)
            
            if value:
                # Process the cached embedding
                if self.compression:
                    value = self._decompress(value)
                embedding = np.frombuffer(value, dtype=np.float32)
                
                # Update metrics
                self.hits += 1
                elapsed = time.time() - start_time
                self.total_saved_time += elapsed
                
                return embedding
            
            # Cache miss
            self.misses += 1
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving from Redis cache: {str(e)}")
            return None
    
    async def get_many(self, keys: List[str]) -> Dict[str, Optional[np.ndarray]]:
        """Batch retrieve multiple embeddings from cache.
        
        Args:
            keys: List of keys to retrieve
            
        Returns:
            Dictionary of {key: embedding} pairs, with None for missing keys
        """
        if not keys:
            return {}
            
        try:
            formatted_keys = [self._format_key(key) for key in keys]
            pipe = self.redis.pipeline()
            for key in formatted_keys:
                pipe.get(key)
            
            results = await pipe.execute()
            
            # Process results
            embeddings = {}
            for i, (key, value) in enumerate(zip(keys, results)):
                if value:
                    # Process the cached embedding
                    if self.compression:
                        value = self._decompress(value)
                    embedding = np.frombuffer(value, dtype=np.float32)
                    embeddings[key] = embedding
                    self.hits += 1
                else:
                    embeddings[key] = None
                    self.misses += 1
                    
            return embeddings
            
        except Exception as e:
            logger.error(f"Error batch retrieving from Redis cache: {str(e)}")
            return {key: None for key in keys}
        
    async def set(self, key: str, embedding: np.ndarray) -> None:
        """Store a single embedding in cache.
        
        Args:
            key: The cache key
            embedding: Numpy array to store
            
        Raises:
            TypeError: If embedding is not a numpy array
        """
        if not isinstance(embedding, np.ndarray):
            raise TypeError("Embedding must be a numpy array")
            
        try:
            formatted_key = self._format_key(key)
            
            # Prepare the embedding data
            data = embedding.astype(np.float32).tobytes()
            if self.compression:
                data = self._compress(data)
                
            await self.redis.set(
                formatted_key,
                data,
                ex=self.ttl
            )
            
        except Exception as e:
            logger.error(f"Error setting Redis cache: {str(e)}")
            raise  # Re-raise the exception
    
    async def set_many(self, items: Dict[str, np.ndarray]) -> None:
        """Batch store multiple embeddings in cache.
        
        Args:
            items: Dictionary of {key: embedding} pairs to store
        """
        if not items:
            return
            
        try:
            # Process in batches to avoid huge pipelines
            keys = list(items.keys())
            for i in range(0, len(keys), self.max_pipeline_size):
                batch_keys = keys[i:i+self.max_pipeline_size]
                pipe = self.redis.pipeline()
                
                for key in batch_keys:
                    embedding = items[key]
                    if not isinstance(embedding, np.ndarray):
                        logger.warning(f"Skipping non-numpy embedding for key {key}")
                        continue
                        
                    formatted_key = self._format_key(key)
                    
                    # Prepare the embedding data
                    data = embedding.astype(np.float32).tobytes()
                    if self.compression:
                        data = self._compress(data)
                        
                    pipe.set(formatted_key, data, ex=self.ttl)
                
                await pipe.execute()
                
        except Exception as e:
            logger.error(f"Error batch setting Redis cache: {str(e)}")
    
    async def delete(self, key: str) -> None:
        """Remove a single key from cache."""
        try:
            formatted_key = self._format_key(key)
            await self.redis.delete(formatted_key)
        except Exception as e:
            logger.error(f"Error deleting from Redis cache: {str(e)}")
    
    async def flush_namespace(self) -> None:
        """Clear all keys in the current namespace."""
        try:
            # Scan and delete all keys in the namespace
            cur = b'0'
            pattern = f"{self.namespace}*"
            while cur:
                cur, keys = await self.redis.scan(cur, match=pattern, count=1000)
                if keys:
                    await self.redis.delete(*keys)
            logger.info(f"Flushed all keys in namespace '{self.namespace}'")
        except Exception as e:
            logger.error(f"Error flushing namespace: {str(e)}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests) if total_requests else 0
        
        # Get memory usage if possible
        try:
            memory_info = await self.redis.info("memory")
            memory_usage = memory_info.get("used_memory_human", "Unknown")
        except:
            memory_usage = "Unknown"
            
        # Count keys in namespace
        try:
            count = 0
            cur = b'0'
            pattern = f"{self.namespace}*"
            while cur:
                cur, keys = await self.redis.scan(cur, match=pattern, count=1000)
                count += len(keys)
        except:
            count = -1
            
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "cached_keys": count,
            "uptime": time.time() - self._start_time,
            "total_saved_time": self.total_saved_time,
            "memory_usage": memory_usage,
            "namespace": self.namespace,
            "compression": self.compression
        }
    
    async def find_similar(self, query_embedding: np.ndarray, top_k: int = 5, threshold: float = 0.8) -> List[Tuple[str, float]]:
        """Find similar embeddings in cache using cosine similarity.
        
        This is a basic implementation that scans all keys.
        For production use with large datasets, consider using:
        - Redis Stack with VEARCH module
        - A separate vector database solution
        
        Args:
            query_embedding: Query embedding to compare against
            top_k: Number of results to return
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of (key, score) tuples sorted by similarity score
        """
        try:
            # Normalize query embedding for cosine similarity
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            
            # Scan all keys in namespace
            results = []
            cur = b'0'
            pattern = f"{self.namespace}*"
            
            while cur:
                cur, keys = await self.redis.scan(cur, match=pattern, count=1000)
                if not keys:
                    continue
                    
                # Get embeddings for these keys
                pipe = self.redis.pipeline()
                for key in keys:
                    pipe.get(key)
                values = await pipe.execute()
                
                # Calculate similarities
                for key, value in zip(keys, values):
                    if value:
                        try:
                            if self.compression:
                                value = self._decompress(value)
                            stored_embedding = np.frombuffer(value, dtype=np.float32)
                            stored_norm = stored_embedding / np.linalg.norm(stored_embedding)
                            
                            # Cosine similarity
                            similarity = np.dot(query_norm, stored_norm)
                            
                            if similarity >= threshold:
                                # Remove namespace prefix to get original key
                                original_key = key.decode('utf-8').replace(self.namespace, '', 1)
                                results.append((original_key, float(similarity)))
                        except Exception as e:
                            logger.warning(f"Error processing cached embedding: {str(e)}")
            
            # Sort and return top_k results
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    async def close(self) -> None:
        """Close Redis connection."""
        try:
            await self.redis.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {str(e)}")


# Convenience factory function
async def create_clip_cache(config: Dict[str, Any]) -> RedisEmbeddingCache:
    """Create and initialize a RedisEmbeddingCache instance."""
    cache = RedisEmbeddingCache(config)
    return cache