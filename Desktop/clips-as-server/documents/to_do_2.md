I've completed the inference pipeline with all necessary functionality. Here's what I've added:
encode_image_from_urls: Asynchronous method to generate embeddings from image URLs
Includes caching
Handles batch processing
Supports timeouts for URL fetching
encode_image_from_bytes: Method to generate embeddings from image bytes
Direct processing of image data
Batch support
Performance logging
_process_text_batch and _process_image_batch: Internal methods for batch processing
Proper device management
Memory-efficient processing with torch.no_grad()
Optional embedding normalization
_fetch_images: Asynchronous image fetching utility
Parallel downloading with aiohttp
Error handling
Timeout support
Cache management utilities:
clear_cache
get_cache_size
warmup: Model warmup functionality
Initializes GPU memory
Runs dummy inference for both text and images
Configurable number of warmup runs
The pipeline is now ready to handle:
Batch processing
Caching
Async operations
Error handling
Performance monitoring
Memory efficiency
Device management
You can use this pipeline by instantiating the InferencePipeline class with your CLIP model and configuration