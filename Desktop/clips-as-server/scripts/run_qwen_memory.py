#!/usr/bin/env python3
"""
Integration script for connecting optimized embeddings to Qwen-VL reasoning.
This script loads NumPy embeddings and metadata, formats them for Qwen-VL,
and runs the Qwen-VL reasoning pipeline to enrich the metadata with visual reasoning.
It serves as a drop-in replacement for run_llava_memory.py using Qwen-VL instead.

Usage examples:
1. Process embeddings and store in Qdrant:
python scripts/run_qwen_memory.py --embeddings-dir ./optimized_comparison_results/embeddings --use-demo-images --upload-to-qdrant

2. Use a custom Qdrant server:
python scripts/run_qwen_memory.py --embeddings-dir ./optimized_comparison_results/embeddings --upload-to-qdrant --qdrant-host my-qdrant-server.com --qdrant-port 6334

3. Create a new collection with a custom name:
python scripts/run_qwen_memory.py --embeddings-dir ./optimized_comparison_results/embeddings --upload-to-qdrant --qdrant-collection qwen_visual_memory --recreate-collection
"""

import os
import sys
import argparse
import numpy as np
import pickle
import json
from pathlib import Path
import torch
from PIL import Image
import logging
from typing import List, Dict, Any, Optional

# Add parent directory to path to import required modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import Qwen-VL utilities
from qwen_vl_utils import QwenVLProcessor, PROMPT_TEMPLATES

# Add import for Qdrant client
from qdrant_client import QdrantClient, models
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("qwen_vl_memory.log")
    ]
)
logger = logging.getLogger(__name__)

def load_embeddings_and_metadata(embeddings_dir: str) -> Dict[str, Any]:
    """
    Load embeddings and metadata from the specified directory.
    
    Args:
        embeddings_dir: Directory containing embeddings and metadata
        
    Returns:
        Dictionary with loaded embeddings and metadata
    """
    logger.info(f"Loading embeddings and metadata from {embeddings_dir}")
    
    # Define paths
    embeddings_dir = Path(embeddings_dir)
    
    # Try to load metadata from pickle file first
    metadata_pkl_path = embeddings_dir / "metadata.pkl"
    metadata_json_path = embeddings_dir / "metadata.json"
    
    # Load metadata
    if metadata_pkl_path.exists():
        with open(metadata_pkl_path, "rb") as f:
            metadata = pickle.load(f)
        logger.info(f"Loaded metadata from {metadata_pkl_path}")
    elif metadata_json_path.exists():
        with open(metadata_json_path, "r") as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata from {metadata_json_path}")
    else:
        raise FileNotFoundError(f"No metadata found in {embeddings_dir}")
    
    # Load large model embeddings (preferred for better reasoning)
    large_img_path = embeddings_dir / "large_img_embeddings.npy"
    large_txt_path = embeddings_dir / "large_txt_embeddings.npy"
    
    if large_img_path.exists() and large_txt_path.exists():
        img_embeddings = np.load(large_img_path)
        txt_embeddings = np.load(large_txt_path)
        model_name = "large"
        logger.info(f"Loaded large model embeddings with shape {img_embeddings.shape}")
    else:
        # Fall back to base model embeddings
        base_img_path = embeddings_dir / "base_img_embeddings.npy"
        base_txt_path = embeddings_dir / "base_txt_embeddings.npy"
        
        if base_img_path.exists() and base_txt_path.exists():
            img_embeddings = np.load(base_img_path)
            txt_embeddings = np.load(base_txt_path)
            model_name = "base"
            logger.info(f"Loaded base model embeddings with shape {img_embeddings.shape}")
        else:
            raise FileNotFoundError(f"No embeddings found in {embeddings_dir}")
    
    # Extract captions from metadata
    if isinstance(metadata, dict) and "captions" in metadata:
        captions = metadata["captions"]
    else:
        # Handle case where metadata is already a list
        captions = [entry.get("caption", "") for entry in metadata] if isinstance(metadata, list) else []
        
    # Get image paths from metadata if available
    if isinstance(metadata, dict) and "image_paths" in metadata:
        image_paths = metadata["image_paths"]
    else:
        # Default to empty paths if not found
        image_paths = ["" for _ in range(len(img_embeddings))]
    
    return {
        "img_embeddings": img_embeddings,
        "txt_embeddings": txt_embeddings,
        "captions": captions,
        "image_paths": image_paths,
        "metadata": metadata,
        "model_name": model_name
    }

def prepare_metadata_for_qwen(data: Dict[str, Any], image_base_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Prepare metadata in the format expected by Qwen-VL.
    
    Args:
        data: Dictionary with embeddings and metadata
        image_base_dir: Base directory for images (if paths are relative)
        
    Returns:
        List of metadata dictionaries suitable for Qwen-VL
    """
    logger.info("Preparing metadata for Qwen-VL reasoning")
    
    metadata_list = []
    
    # Determine number of samples
    num_samples = len(data["img_embeddings"])
    
    for i in range(num_samples):
        # Create metadata entry
        entry = {
            "id": f"sample_{i}",
            "embedding_id": i,  # Store the index in the embedding array
            "image_path": data["image_paths"][i] if i < len(data["image_paths"]) else "",
            "caption": data["captions"][i] if i < len(data["captions"]) else "",
            f"{data['model_name']}_img_embedding": data["img_embeddings"][i].tolist() if isinstance(data["img_embeddings"][i], np.ndarray) else data["img_embeddings"][i],
            f"{data['model_name']}_txt_embedding": data["txt_embeddings"][i].tolist() if isinstance(data["txt_embeddings"][i], np.ndarray) else data["txt_embeddings"][i],
        }
        
        # Handle image paths
        if entry["image_path"] and image_base_dir:
            # Check if path is relative or absolute
            if not os.path.isabs(entry["image_path"]):
                entry["image_path"] = os.path.join(image_base_dir, entry["image_path"])
        
        metadata_list.append(entry)
    
    logger.info(f"Prepared {len(metadata_list)} metadata entries for Qwen-VL")
    return metadata_list

def run_qwen_on_metadata(metadata_list: List[Dict[str, Any]], 
                         qwen_model: str = "Qwen/Qwen-VL-Chat",
                         use_demo_images: bool = False,
                         demo_images_dir: str = "demo_images") -> List[Dict[str, Any]]:
    """
    Run Qwen-VL reasoning on metadata.
    
    Args:
        metadata_list: List of metadata dictionaries
        qwen_model: Qwen-VL model name
        use_demo_images: Whether to use demo images when image_path doesn't exist
        demo_images_dir: Directory containing demo images
        
    Returns:
        Enriched metadata with Qwen-VL reasoning
    """
    logger.info(f"Running Qwen-VL reasoning using model: {qwen_model}")
    
    # Initialize Qwen-VL processor
    qwen = QwenVLProcessor(model_name=qwen_model)
    
    # Define prompts for Qwen-VL reasoning
    prompts = {
        "qwen_caption": PROMPT_TEMPLATES["caption"],
        "qwen_scene": PROMPT_TEMPLATES["scene"],
        "qwen_table": PROMPT_TEMPLATES["table"],
        "qwen_qa": PROMPT_TEMPLATES["qa"],
        "qwen_explanation": PROMPT_TEMPLATES["explanation"]
    }
    
    # Create demo images directory if needed
    if use_demo_images:
        os.makedirs(demo_images_dir, exist_ok=True)
        
        # Create colored squares for demo
        demo_images = {}
        for color in ["red", "blue", "green", "yellow", "purple"]:
            img = Image.new('RGB', (224, 224), color=color)
            path = os.path.join(demo_images_dir, f"{color}_square.png")
            img.save(path)
            demo_images[color] = path
    
    # Process each metadata entry
    updated_metadata = []
    
    for i, entry in enumerate(metadata_list):
        logger.info(f"Processing entry {i+1}/{len(metadata_list)}")
        
        # Check if image path exists
        image_path = entry.get("image_path", "")
        valid_image = image_path and os.path.exists(image_path)
        
        if not valid_image and use_demo_images:
            # Use demo image as fallback
            # Cycle through demo colors
            colors = list(demo_images.keys())
            color = colors[i % len(colors)]
            image_path = demo_images[color]
            logger.info(f"Using demo image {image_path}")
            entry["image_path"] = image_path
            valid_image = True
        
        # Run Qwen-VL reasoning if image is valid
        if valid_image:
            try:
                # Load image
                img = Image.open(image_path).convert("RGB")
                
                # Apply each prompt
                for key, prompt in prompts.items():
                    entry[key] = qwen.infer(img, prompt)
                    
                logger.info(f"Successfully processed reasoning for {image_path}")
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
                for key in prompts.keys():
                    entry[key] = f"Error: {str(e)}"
        else:
            logger.warning(f"No valid image found for entry {i+1}")
            # Add placeholder responses
            for key in prompts.keys():
                entry[key] = "No image available for processing"
        
        updated_metadata.append(entry)
    
    logger.info(f"Completed Qwen-VL reasoning on {len(updated_metadata)} entries")
    return updated_metadata

def store_in_qdrant(
    enriched_metadata: List[Dict[str, Any]],
    host: str = "localhost",
    port: int = 6333,
    collection_name: str = "qwen_multimodal_memory",
    recreate_collection: bool = False
) -> None:
    """
    Store the enriched metadata in Qdrant vector database.
    
    Args:
        enriched_metadata: List of metadata dictionaries with embeddings and Qwen-VL reasoning
        host: Qdrant host address
        port: Qdrant port
        collection_name: Name of the collection to store data in
        recreate_collection: Whether to recreate the collection if it exists
    """
    logger.info(f"Connecting to Qdrant at {host}:{port}")
    
    # Connect to Qdrant
    client = QdrantClient(host, port=port)
    
    # Determine embedding dimension from data
    first_entry = enriched_metadata[0]
    
    # Check which embeddings we have
    embedding_types = {}
    for key in first_entry:
        if key.endswith("_img_embedding") and isinstance(first_entry[key], (list, np.ndarray)):
            embed_type = key.replace("_img_embedding", "")
            embed_dim = len(first_entry[key])
            embedding_types[embed_type] = embed_dim
            logger.info(f"Found {embed_type} embeddings with dimension {embed_dim}")
    
    if not embedding_types:
        logger.error("No embeddings found in metadata. Cannot upload to Qdrant.")
        return
    
    # Create or recreate collection
    if recreate_collection:
        logger.info(f"Recreating collection '{collection_name}'")
        
        # Prepare vector configurations for all embedding types
        vectors_config = {}
        for embed_type, dim in embedding_types.items():
            vectors_config[embed_type] = models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE
            )
        
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=vectors_config
        )
    else:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_exists = any(c.name == collection_name for c in collections)
        
        if not collection_exists:
            logger.info(f"Creating new collection '{collection_name}'")
            
            # Prepare vector configurations for all embedding types
            vectors_config = {}
            for embed_type, dim in embedding_types.items():
                vectors_config[embed_type] = models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE
                )
            
            client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config
            )
    
    # Upload data to Qdrant in batches
    batch_size = 100
    total_entries = len(enriched_metadata)
    
    logger.info(f"Uploading {total_entries} entries to Qdrant in batches of {batch_size}")
    
    for batch_start in range(0, total_entries, batch_size):
        batch_end = min(batch_start + batch_size, total_entries)
        batch = enriched_metadata[batch_start:batch_end]
        
        # Prepare batch for upload
        points = []
        
        for entry in batch:
            # Generate a unique ID if not present
            if "id" not in entry or not entry["id"]:
                entry["id"] = str(uuid.uuid4())
            
            # Extract vectors for each embedding type
            vectors = {}
            for embed_type in embedding_types:
                img_key = f"{embed_type}_img_embedding"
                if img_key in entry:
                    # Convert to list if it's a numpy array
                    if isinstance(entry[img_key], np.ndarray):
                        vectors[embed_type] = entry[img_key].tolist()
                    else:
                        vectors[embed_type] = entry[img_key]
            
            # Create a copy of the payload without the embeddings
            payload = {k: v for k, v in entry.items() if not k.endswith("_embedding")}
            
            # Add to points list
            points.append(
                models.PointStruct(
                    id=entry["id"],
                    vector=vectors,
                    payload=payload
                )
            )
        
        # Upload batch to Qdrant
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        logger.info(f"Uploaded batch {batch_start//batch_size + 1}/{(total_entries-1)//batch_size + 1} ({batch_end}/{total_entries} entries)")
    
    logger.info(f"Successfully uploaded {total_entries} entries to Qdrant collection '{collection_name}'")

def search_similar_images(
    query_embedding: List[float],
    embedding_type: str = "large",
    host: str = "localhost",
    port: int = 6333,
    collection_name: str = "qwen_multimodal_memory",
    limit: int = 5,
    filters: Optional[Dict] = None
) -> List[Dict[str, Any]]:
    """
    Search for similar images in Qdrant based on embedding.
    
    Args:
        query_embedding: The embedding to search with
        embedding_type: Type of embedding to use (e.g., 'large', 'base')
        host: Qdrant host address
        port: Qdrant port
        collection_name: Name of the collection to search in
        limit: Maximum number of results to return
        filters: Optional filters to apply to the search
        
    Returns:
        List of matching results with scores and payloads
    """
    logger.info(f"Searching for similar images in Qdrant using {embedding_type} embeddings")
    
    # Connect to Qdrant
    client = QdrantClient(host, port=port)
    
    # Prepare filters if provided
    qdrant_filter = None
    if filters:
        must_conditions = []
        for key, value in filters.items():
            must_conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value)
                )
            )
        
        if must_conditions:
            qdrant_filter = models.Filter(
                must=must_conditions
            )
    
    # Run the search
    search_results = client.search(
        collection_name=collection_name,
        query_vector={embedding_type: query_embedding},
        query_filter=qdrant_filter,
        limit=limit,
        with_payload=True
    )
    
    # Format results
    results = []
    for hit in search_results:
        result = {
            "id": hit.id,
            "score": hit.score,
            **hit.payload
        }
        results.append(result)
    
    logger.info(f"Found {len(results)} similar images")
    return results

def main():
    parser = argparse.ArgumentParser(description="Connect embeddings to Qwen-VL reasoning with memory")
    parser.add_argument("--embeddings-dir", type=str, 
                      default="/Users/songxiaomei/Desktop/meta-MLE-interview/clip-service/optimized_comparison_results/embeddings",
                      help="Directory containing embeddings and metadata")
    parser.add_argument("--output-file", type=str, default="qwen_vl_enriched_metadata.pkl",
                      help="Output file for enriched metadata")
    parser.add_argument("--image-base-dir", type=str, default=None,
                      help="Base directory for images (if paths are relative)")
    parser.add_argument("--qwen-model", type=str, default="Qwen/Qwen-VL-Chat",
                      help="Qwen-VL model name")
    parser.add_argument("--use-demo-images", action="store_true",
                      help="Use demo images when image path doesn't exist")
    
    # Add Qdrant-related arguments
    parser.add_argument("--upload-to-qdrant", action="store_true",
                      help="Upload enriched metadata to Qdrant")
    parser.add_argument("--qdrant-host", type=str, default="localhost",
                      help="Qdrant host address")
    parser.add_argument("--qdrant-port", type=int, default=6333,
                      help="Qdrant port")
    parser.add_argument("--qdrant-collection", type=str, default="qwen_multimodal_memory",
                      help="Qdrant collection name")
    parser.add_argument("--recreate-collection", action="store_true",
                      help="Recreate Qdrant collection if it exists")
    
    args = parser.parse_args()
    
    # Load embeddings and metadata
    data = load_embeddings_and_metadata(args.embeddings_dir)
    
    # Prepare metadata for Qwen-VL
    metadata_list = prepare_metadata_for_qwen(data, args.image_base_dir)
    
    # Run Qwen-VL reasoning
    enriched_metadata = run_qwen_on_metadata(
        metadata_list=metadata_list,
        qwen_model=args.qwen_model,
        use_demo_images=args.use_demo_images
    )
    
    # Save enriched metadata
    with open(args.output_file, "wb") as f:
        pickle.dump(enriched_metadata, f)
    
    # Also save as JSON for easier inspection
    json_output = args.output_file.replace(".pkl", ".json")
    try:
        with open(json_output, "w") as f:
            json.dump(enriched_metadata, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save JSON version: {e}")
    
    logger.info(f"Saved enriched metadata to {args.output_file} and {json_output}")
    
    # Upload to Qdrant if requested
    if args.upload_to_qdrant:
        try:
            store_in_qdrant(
                enriched_metadata=enriched_metadata,
                host=args.qdrant_host,
                port=args.qdrant_port,
                collection_name=args.qdrant_collection,
                recreate_collection=args.recreate_collection
            )
        except Exception as e:
            logger.error(f"Error uploading to Qdrant: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 