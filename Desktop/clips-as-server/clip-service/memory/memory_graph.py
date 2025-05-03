"""Memory Graph implementation for CLIP service."""
from typing import List, Any, Dict
import redis
from qdrant_client import QdrantClient

class MemoryGraph:
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, 
                 qdrant_url: str = "http://localhost:6333"):
        # Initialize Redis client
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = "memory_graph"
        
        # Ensure collection exists
        self._init_collection()
    
    def _init_collection(self) -> None:
        """Initialize Qdrant collection if it doesn't exist."""
        collections = self.qdrant_client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.qdrant_client.create_collection(self.collection_name)
    
    def add_memory(self, user_id: str, memory_data: Any) -> None:
        """Add memory to Redis."""
        self.redis_client.lpush(f"user:{user_id}:memory", memory_data)
    
    def get_memory(self, user_id: str) -> List[bytes]:
        """Retrieve memory from Redis."""
        return self.redis_client.lrange(f"user:{user_id}:memory", 0, -1)

# Example usage
if __name__ == "__main__":
    # Add memory
    memory_graph = MemoryGraph()
    memory_graph.add_memory("user1", "This is a test memory")
    memory_graph.add_memory("user1", "Another test memory")

    # Retrieve memory
    print(memory_graph.get_memory("user1"))      