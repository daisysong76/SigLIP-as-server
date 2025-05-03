# Sharded FAISS index with GPU acceleration
class DistributedVectorIndex:
    def __init__(self, dimensions, num_shards=8):
        self.dimensions = dimensions
        self.num_shards = num_shards
        self.shards = []
        self.shard_mapping = {}
        
        # Initialize shards across available GPUs
        for i in range(num_shards):
            gpu_id = i % torch.cuda.device_count()
            self.shards.append(self._create_gpu_index(gpu_id))
            
    def _create_gpu_index(self, gpu_id):
        # Create highly optimized HNSW index with PQ compression
        cpu_index = faiss.IndexHNSWPQ(
            self.dimensions,  # Dimensionality 
            96,               # PQ sub-vectors
            8,                # Bits per sub-vector
            32                # HNSW neighbors
        )
        cpu_index.hnsw.efConstruction = 128
        cpu_index.train(representative_data)
        
        # Move to GPU
        return faiss.index_cpu_to_gpu(
            faiss.StandardGpuResources(),
            gpu_id,
            cpu_index
        )
        
    def add_vectors(self, vectors, ids):
        # Consistent sharding using xxHash
        for i, (vec, id) in enumerate(zip(vectors, ids)):
            shard_id = xxhash.xxh64(id.encode()).intdigest() % self.num_shards
            self.shards[shard_id].add_with_ids(vec.reshape(1, -1), np.array([id]))
            self.shard_mapping[id] = shard_id
            
    async def search(self, query_vector, k=10):
        # Search across all shards in parallel
        shard_futures = []
        for shard_id, shard in enumerate(self.shards):
            shard_futures.append(
                asyncio.to_thread(
                    self._search_shard,
                    shard_id,
                    query_vector,
                    k
                )
            )
        
        # Gather and merge results
        results = await asyncio.gather(*shard_futures)
        merged_distances = np.concatenate([r[0] for r in results])
        merged_indices = np.concatenate([r[1] for r in results])
        
        # Sort and return top-k
        sorted_indices = np.argsort(merged_distances)[:k]
        return merged_distances[sorted_indices], merged_indices[sorted_indices]