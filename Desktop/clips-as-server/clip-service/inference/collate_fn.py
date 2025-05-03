"""
Advanced collate functions for multimodal data batching.
Handles scalar IDs with proper tensor conversion following best practices from top ML labs.

IMPORTANT ARCHITECTURAL NOTES:
1. Primary entity IDs are kept as scalars within individual samples for fast retrieval
2. When batched, IDs are collected as tensors with plural names (image_ids, text_ids) for clarity
3. We use int64 tensors for IDs to optimize CPU-GPU transfer and database operations
4. Embeddings and IDs are kept separate, allowing for efficient post-processing
5. This approach aligns with industry practices at DeepMind, OpenAI, and Meta
"""
import torch
from typing import List, Dict, Any, Optional, Union, Tuple


def multimodal_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for batching multimodal (text + image) samples.
    Follows industry best practices for handling IDs at scale:
    - Scalar IDs for fast retrieval
    - Minimal padding
    - Separate batch metadata from embeddings
    - Efficient CPU-GPU transfer with int64 tensors for IDs
    
    Args:
        batch: List of sample dictionaries to collate
        
    Returns:
        Dict containing batched tensors with appropriate keys
    """
    # Initialize collections
    pixel_values = []
    input_ids = []
    attention_masks = []
    image_ids = []
    text_ids = []
    
    # Extract items from each sample
    for sample in batch:
        if "pixel_values" in sample:
            pixel_values.append(sample["pixel_values"])
        if "input_ids" in sample:
            input_ids.append(sample["input_ids"])
        if "attention_mask" in sample:
            attention_masks.append(sample["attention_mask"])
        if "image_id" in sample:
            image_ids.append(sample["image_id"])
        if "text_id" in sample:
            text_ids.append(sample["text_id"])
    
    # Build result dictionary with batched tensors
    result = {}
    
    # Stack image tensors - model inputs for embedding generation
    if pixel_values:
        result["pixel_values"] = torch.stack(pixel_values)
    
    # Stack text tensors - model inputs for embedding generation
    if input_ids:
        result["input_ids"] = torch.stack(input_ids)
    if attention_masks:
        result["attention_mask"] = torch.stack(attention_masks)
    
    # Convert ID lists to efficient tensors - kept separate as batch metadata
    # Using torch.int64 for most efficient CPU-GPU transfer and compatibility with indexes
    if image_ids:
        try:
            # Keep IDs as int64 tensors for fast transfer and db operations
            result["image_ids"] = torch.tensor(image_ids, dtype=torch.int64)  # Plurality for batch-level metadata
        except (ValueError, TypeError):
            # Fallback for non-numeric IDs or mixed types
            result["image_ids"] = image_ids
    
    if text_ids:
        try:
            # Keep IDs as int64 tensors for fast transfer and db operations
            result["text_ids"] = torch.tensor(text_ids, dtype=torch.int64)  # Plurality for batch-level metadata
        except (ValueError, TypeError):
            # Fallback for non-numeric IDs or mixed types
            result["text_ids"] = text_ids
    
    return result


def image_only_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function optimized for image-only batches.
    Use when processing only images with CLIP.
    
    Args:
        batch: List of sample dictionaries to collate
        
    Returns:
        Dict containing batched image tensors and IDs
    """
    pixel_values = []
    image_ids = []
    
    for sample in batch:
        if "pixel_values" in sample:
            pixel_values.append(sample["pixel_values"])
        if "image_id" in sample:
            image_ids.append(sample["image_id"])
    
    result = {}
    
    # Model inputs
    if pixel_values:
        result["pixel_values"] = torch.stack(pixel_values)
    
    # Batch metadata
    if image_ids:
        try:
            # Use int64 for efficient CPU-GPU transfer and compatibility
            result["image_ids"] = torch.tensor(image_ids, dtype=torch.int64)
        except (ValueError, TypeError):
            result["image_ids"] = image_ids
    
    return result


def text_only_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function optimized for text-only batches.
    Use when processing only text with CLIP.
    
    Args:
        batch: List of sample dictionaries to collate
        
    Returns:
        Dict containing batched text tensors and IDs
    """
    input_ids = []
    attention_masks = []
    text_ids = []
    
    for sample in batch:
        if "input_ids" in sample:
            input_ids.append(sample["input_ids"])
        if "attention_mask" in sample:
            attention_masks.append(sample["attention_mask"])
        if "text_id" in sample:
            text_ids.append(sample["text_id"])
    
    result = {}
    
    # Model inputs
    if input_ids:
        result["input_ids"] = torch.stack(input_ids)
    if attention_masks:
        result["attention_mask"] = torch.stack(attention_masks)
    
    # Batch metadata
    if text_ids:
        try:
            # Use int64 for efficient CPU-GPU transfer and compatibility
            result["text_ids"] = torch.tensor(text_ids, dtype=torch.int64)
        except (ValueError, TypeError):
            result["text_ids"] = text_ids
    
    return result


def postprocess_embeddings(
    embeddings: torch.Tensor, 
    ids: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Post-process embeddings and IDs for retrieval tasks.
    Follows industry best practices for handling embeddings and IDs after model outputs.
    
    Args:
        embeddings: Tensor of shape [batch_size, embedding_dim]
        ids: Tensor of IDs corresponding to each embedding
        
    Returns:
        Tuple of (normalized_embeddings, ids)
    """
    # Normalize embeddings for cosine similarity
    normalized = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    
    # Ensure IDs are the right type and shape
    if not isinstance(ids, torch.Tensor):
        ids = torch.tensor(ids, dtype=torch.int64)
    
    return normalized, ids


def retrieve_top_k(
    query_embedding: torch.Tensor,
    index_embeddings: torch.Tensor,
    index_ids: torch.Tensor,
    k: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Retrieve top-k similar items using efficient ID handling.
    
    Args:
        query_embedding: Query embedding of shape [embedding_dim]
        index_embeddings: Index embeddings of shape [index_size, embedding_dim]
        index_ids: IDs for each index embedding
        k: Number of results to return
        
    Returns:
        Tuple of (similarity_scores, retrieved_ids)
    """
    # Compute similarities
    similarities = torch.matmul(index_embeddings, query_embedding)
    
    # Get top-k
    if k < similarities.shape[0]:
        values, indices = torch.topk(similarities, k)
    else:
        values, indices = torch.sort(similarities, descending=True)
        values = values[:k]
        indices = indices[:k]
    
    # Gather corresponding IDs - using IDs for post-processing
    retrieved_ids = torch.gather(index_ids, 0, indices)
    
    return values, retrieved_ids 