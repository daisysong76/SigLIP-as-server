#!/usr/bin/env python3
#1. Process embeddings and store in Qdrant:
#python scripts/run_llava_memory.py --embeddings-dir ./optimized_comparison_results/embeddings --use-demo-images --upload-to-qdrant
#2. Use a custom Qdrant server:
#python scripts/run_llava_memory.py --embeddings-dir ./optimized_comparison_results/embeddings --upload-to-qdrant --qdrant-host my-qdrant-server.com --qdrant-port 6334
#3. Create a new collection with a custom name:
#python scripts/run_llava_memory.py --embeddings-dir ./optimized_comparison_results/embeddings --upload-to-qdrant --qdrant-collection my_custom_collection

"""
Integration script for connecting optimized embeddings to LLaVA reasoning.
This script loads NumPy embeddings and metadata, formats them for LLaVA,
and runs the LLaVA reasoning pipeline to enrich the metadata with visual reasoning.
store_in_qdrant(): Uploads enriched metadata to Qdrant
Automatically detects and configures embedding types
Creates collections with the right vector dimensions
Handles batched uploads for efficiency
Stores all LLaVA reasoning outputs and metadata as searchable fields
search_similar_images(): Retrieves similar images from Qdrant
Searches using embeddings from any model type
Supports filtering by metadata fields
Returns complete results with similarity scores

Loads the LLaVA model (defaults to "llava-hf/llava-1.5-7b-hf")
Processes images with multiple prompt templates for different tasks:
Caption generation
Scene description
Structured table information
Question answering
Explanations

Benefits of the Integration:
Complete Memory System: Your embeddings and LLaVA reasoning are now stored in a searchable database
Multimodal Vector Search: You can search by visual similarity using any model type
Rich Metadata Filtering: Filter by captions, LLaVA outputs, or any other fields
Scalable Architecture: Works with large datasets through batched processing
Future Integration: Ready for integration with agent systems that need visual memory

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

# Import LLaVA utilities
from llava_utils import LLaVAProcessor, PROMPT_TEMPLATES

# Add import for Qdrant client
from qdrant_client import QdrantClient, models
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("embed_to_llava.log")
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

def prepare_metadata_for_llava(data: Dict[str, Any], image_base_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Prepare metadata in the format expected by LLaVA.
    
    Args:
        data: Dictionary with embeddings and metadata
        image_base_dir: Base directory for images (if paths are relative)
        
    Returns:
        List of metadata dictionaries suitable for LLaVA
    """
    logger.info("Preparing metadata for LLaVA reasoning")
    
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
    
    logger.info(f"Prepared {len(metadata_list)} metadata entries for LLaVA")
    return metadata_list

def run_llava_on_metadata(metadata_list: List[Dict[str, Any]], 
                         llava_model: str = "llava-hf/llava-1.5-7b-hf",
                         use_demo_images: bool = False,
                         demo_images_dir: str = "demo_images") -> List[Dict[str, Any]]:
    """
    Run LLaVA reasoning on metadata.
    
    Args:
        metadata_list: List of metadata dictionaries
        llava_model: LLaVA model name
        use_demo_images: Whether to use demo images when image_path doesn't exist
        demo_images_dir: Directory containing demo images
        
    Returns:
        Enriched metadata with LLaVA reasoning
    """
    logger.info(f"Running LLaVA reasoning using model: {llava_model}")
    
    # Initialize LLaVA processor
    try:
        llava = LLaVAProcessor(model_name=llava_model)
    except Exception as e:
        logger.error(f"Failed to initialize LLaVA model: {e}")
        # Provide empty results if model fails to load
        for entry in metadata_list:
            for key in ["llava_caption", "llava_scene", "llava_table", "llava_qa", "llava_explanation"]:
                entry[key] = f"Error loading LLaVA model: {str(e)}"
        return metadata_list
    
    # Define prompts for LLaVA reasoning
    prompts = {
        "llava_caption": PROMPT_TEMPLATES["caption"],
        "llava_scene": PROMPT_TEMPLATES["scene"],
        "llava_table": PROMPT_TEMPLATES["table"],
        "llava_qa": PROMPT_TEMPLATES["qa"],
        "llava_explanation": PROMPT_TEMPLATES["explanation"]
    }
    
    # Create demo images directory if needed
    demo_images = {}
    if use_demo_images:
        os.makedirs(demo_images_dir, exist_ok=True)
        
        # Create colored squares for demo
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
        
        # Run LLaVA reasoning if image is valid
        if valid_image:
            for key, prompt in prompts.items():
                try:
                    # Process directly with llava processor - let it handle PIL image loading
                    result = llava.infer(image_path, prompt)
                    entry[key] = result
                    logger.info(f"Successfully processed reasoning for {image_path} with prompt '{key}'")
                except Exception as e:
                    error_msg = f"Error processing image {image_path} with prompt '{key}': {e}"
                    logger.error(error_msg)
                    entry[key] = f"Error: {str(e)}"
        else:
            logger.warning(f"No valid image found for entry {i+1}")
            # Add placeholder responses
            for key in prompts.keys():
                entry[key] = "No image available for processing"
        
        updated_metadata.append(entry)
    
    logger.info(f"Completed LLaVA reasoning on {len(updated_metadata)} entries")
    return updated_metadata

def store_in_qdrant(
    enriched_metadata: List[Dict[str, Any]],
    host: str = "localhost",
    port: int = 6333,
    collection_name: str = "multimodal_memory",
    recreate_collection: bool = False
) -> None:
    """
    Store the enriched metadata in Qdrant vector database.
    
    Args:
        enriched_metadata: List of metadata dictionaries with embeddings and LLaVA reasoning
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
    collection_name: str = "multimodal_memory",
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

def save_results_as_text(enriched_metadata: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save the LLaVA reasoning results to a formatted text file.
    
    Args:
        enriched_metadata: List of metadata entries with LLaVA reasoning
        output_file: Path to the output text file
    """
    logger.info(f"Saving LLaVA reasoning results to text file: {output_file}")
    
    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("LLaVA REASONING RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for i, entry in enumerate(enriched_metadata):
            f.write(f"ENTRY {i+1}/{len(enriched_metadata)}\n")
            f.write("-" * 80 + "\n")
            
            # Write basic info
            f.write(f"ID: {entry.get('id', f'Sample {i}')}\n")
            if 'caption' in entry:
                f.write(f"Caption: {entry['caption']}\n")
            if 'image_path' in entry:
                f.write(f"Image path: {entry['image_path']}\n")
            
            f.write("\n")
            
            # Write all LLaVA results
            llava_keys = [k for k in entry.keys() if k.startswith("llava_")]
            
            if not llava_keys:
                f.write("No LLaVA results found for this entry.\n")
            else:
                for key in llava_keys:
                    display_name = key.replace("llava_", "").upper()
                    f.write(f"{display_name}:\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"{entry[key]}\n\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    
    logger.info(f"Successfully saved results to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Connect embeddings to LLaVA reasoning with memory")
    parser.add_argument("--embeddings-dir", type=str, 
                      default="/Users/songxiaomei/Desktop/meta-MLE-interview/clip-service/optimized_comparison_results/embeddings",
                      help="Directory containing embeddings and metadata")
    parser.add_argument("--output-file", type=str, default="llava_enriched_metadata.pkl",
                      help="Output file for enriched metadata")
    parser.add_argument("--text-output", type=str, default="llava_reasoning.txt",
                      help="Output text file for human-readable LLaVA results")
    parser.add_argument("--image-base-dir", type=str, default=None,
                      help="Base directory for images (if paths are relative)")
    parser.add_argument("--llava-model", type=str, default="llava-hf/llava-1.5-7b-hf",
                      help="LLaVA model name")
    parser.add_argument("--use-demo-images", action="store_true",
                      help="Use demo images when image path doesn't exist")
    
    # Add Qdrant-related arguments
    parser.add_argument("--upload-to-qdrant", action="store_true",
                      help="Upload enriched metadata to Qdrant")
    parser.add_argument("--qdrant-host", type=str, default="localhost",
                      help="Qdrant host address")
    parser.add_argument("--qdrant-port", type=int, default=6333,
                      help="Qdrant port")
    parser.add_argument("--qdrant-collection", type=str, default="multimodal_memory",
                      help="Qdrant collection name")
    parser.add_argument("--recreate-collection", action="store_true",
                      help="Recreate Qdrant collection if it exists")
    
    args = parser.parse_args()
    
    # Load embeddings and metadata
    data = load_embeddings_and_metadata(args.embeddings_dir)
    
    # Prepare metadata for LLaVA
    metadata_list = prepare_metadata_for_llava(data, args.image_base_dir)
    
    # Run LLaVA reasoning
    enriched_metadata = run_llava_on_metadata(
        metadata_list=metadata_list,
        llava_model=args.llava_model,
        use_demo_images=args.use_demo_images
    )
    
    # Save enriched metadata as pickle
    with open(args.output_file, "wb") as f:
        pickle.dump(enriched_metadata, f)
    
    # Save as JSON for easier inspection
    json_output = args.output_file.replace(".pkl", ".json")
    try:
        with open(json_output, "w") as f:
            json.dump(enriched_metadata, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save JSON version: {e}")
    
    logger.info(f"Saved enriched metadata to {args.output_file} and {json_output}")
    
    # Save as formatted text file
    save_results_as_text(enriched_metadata, args.text_output)
    
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