"""
Test script for using LLaVA model with pre-computed CLIP embeddings.
This approach separates image encoding (CLIP) from reasoning (LLaVA).

Usage:
    python scripts/test_llava_embeddings.py --embeddings comparison_results/embeddings
"""
import os
import sys
import argparse
import logging
import torch
import glob
import pickle
import traceback
import time
from tqdm import tqdm
from pathlib import Path
from llava_utils import LLaVAProcessor, PROMPT_TEMPLATES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Smaller, faster LLaVA models
FAST_MODELS = [
    "llava-hf/llava-1.5-7b-hf",     # Original model - large but works
    "llava-hf/llava-v1.6-mistral-7b-hf",  # Mistral-based version - might be faster
]

def load_embeddings(embeddings_dir, limit=10):
    """Load embeddings from a directory containing .pt or .pickle files."""
    logger.info(f"Loading embeddings from: {embeddings_dir}")
    
    embeddings = []
    paths = []
    
    # Try to find pickle files first (may contain metadata)
    pickle_files = sorted(glob.glob(os.path.join(embeddings_dir, "*.pickle")))
    if pickle_files:
        logger.info(f"Found {len(pickle_files)} pickle files")
        file_count = min(len(pickle_files), limit)
        logger.info(f"Loading up to {file_count} pickle files")
        for pickle_file in tqdm(pickle_files[:file_count], desc="Loading pickles"):
            try:
                with open(pickle_file, 'rb') as f:
                    data = pickle.load(f)
                    # Handle different pickle formats
                    if isinstance(data, dict) and "embedding" in data:
                        embeddings.append(data["embedding"])
                        paths.append(data.get("image_path", pickle_file))
                    elif isinstance(data, torch.Tensor):
                        embeddings.append(data)
                        paths.append(pickle_file)
                    else:
                        logger.warning(f"Skipping {pickle_file} with unexpected format")
            except Exception as e:
                logger.error(f"Error loading {pickle_file}: {e}")
    
    # If no pickle files found, try .pt files
    if not embeddings:
        pt_files = sorted(glob.glob(os.path.join(embeddings_dir, "*.pt")))
        if not pt_files:
            logger.error(f"No .pt files found in {embeddings_dir}")
            return []
            
        logger.info(f"Found {len(pt_files)} PT files")
        file_count = min(len(pt_files), limit)
        logger.info(f"Loading up to {file_count} PT files")
        
        # Filter out metadata files which often cause issues
        pt_files = [f for f in pt_files if not "metadata" in f.lower()]
        if not pt_files:
            logger.error("No valid PT files found after filtering")
            return []
            
        for pt_file in tqdm(pt_files[:file_count], desc="Loading tensors"):
            try:
                # Try the safer approach first with weights_only=True
                logger.info(f"Attempting to load {pt_file} with weights_only=True")
                embedding = torch.load(pt_file, weights_only=True, map_location="cpu")
                
                # Process the embedding based on its type
                if isinstance(embedding, dict):
                    # If it's a dict with 'embeddings' key
                    if 'embeddings' in embedding:
                        # Handle case where 'embeddings' is the actual embedding tensor
                        if isinstance(embedding['embeddings'], torch.Tensor):
                            embeddings.append(embedding['embeddings'])
                            paths.append(pt_file)
                        # Handle case where 'embeddings' is a dict of embeddings
                        elif isinstance(embedding['embeddings'], dict):
                            for key, emb in embedding['embeddings'].items():
                                if isinstance(emb, torch.Tensor):
                                    embeddings.append(emb)
                                    paths.append(f"{pt_file}:{key}")
                    # If it's a dictionary with various keys, just use all tensor values
                    else:
                        for key, value in embedding.items():
                            if isinstance(value, torch.Tensor) and len(value.shape) > 0:
                                embeddings.append(value)
                                paths.append(f"{pt_file}:{key}")
                elif isinstance(embedding, torch.Tensor):
                    embeddings.append(embedding)
                    paths.append(pt_file)
                else:
                    logger.warning(f"Unsupported embedding format in {pt_file}: {type(embedding)}")
            except Exception as e:
                logger.error(f"Error loading {pt_file}: {e}")
                # Try alternative loading method with weights_only=False
                try:
                    logger.info(f"Attempting to load {pt_file} with weights_only=False")
                    embedding = torch.load(pt_file, weights_only=False, map_location="cpu")
                    if isinstance(embedding, torch.Tensor):
                        embeddings.append(embedding)
                        paths.append(pt_file)
                    else:
                        logger.warning(f"Skipping {pt_file} with unexpected format: {type(embedding)}")
                except Exception as inner_e:
                    logger.error(f"Still could not load {pt_file}: {inner_e}")
    
    logger.info(f"Successfully loaded {len(embeddings)} embeddings")
    
    # Create sample embedding if none were loaded
    if not embeddings:
        logger.warning("No embeddings loaded. Creating a sample random embedding for testing.")
        sample_embedding = torch.randn(512)  # Standard CLIP embedding size
        embeddings.append(sample_embedding)
        paths.append("random_sample_embedding")
    
    # Create metadata list with embeddings
    metadata = []
    for i, (emb, path) in enumerate(zip(embeddings, paths)):
        # Ensure the embedding is a tensor and has the right shape
        if not isinstance(emb, torch.Tensor):
            logger.warning(f"Skipping non-tensor embedding at {path}: {type(emb)}")
            continue
            
        # Basic shape validation
        if len(emb.shape) == 0:
            logger.warning(f"Skipping scalar tensor at {path}")
            continue
            
        # If tensor is 2D with a single item, squeeze it
        if len(emb.shape) == 2 and (emb.shape[0] == 1 or emb.shape[1] == 1):
            emb = emb.squeeze()
            
        # Add to metadata
        metadata.append({
            "id": i,
            "embedding": emb,
            "source": path,
        })
    
    return metadata

def test_llava_with_embeddings(embeddings_dir, model_name="llava-hf/llava-1.5-7b-hf", max_samples=2, timeout=30):
    """Test LLaVA processor with pre-computed embeddings."""
    logger.info(f"Testing LLaVA with embeddings from: {embeddings_dir}")
    start_time = time.time()
    
    # Load embeddings
    metadata = load_embeddings(embeddings_dir, limit=max_samples)
    if not metadata:
        logger.error(f"No embeddings found in {embeddings_dir}")
        return False
    
    # Limit number of samples for testing
    if len(metadata) > max_samples:
        logger.info(f"Limiting to {max_samples} samples for testing")
        metadata = metadata[:max_samples]
    
    try:
        # Initialize the processor
        logger.info(f"Initializing LLaVA processor with model: {model_name}")
        processor = LLaVAProcessor(model_name=model_name)
        
        # Define test prompts - just use one simple prompt for faster testing
        test_prompts = {
            "caption": "Describe this image briefly.",
        }
        
        # Process each embedding
        success_count = 0
        for i, entry in enumerate(metadata):
            # Check timeout
            if time.time() - start_time > timeout:
                logger.warning(f"Timeout reached after {timeout} seconds. Stopping.")
                break
                
            logger.info(f"Processing embedding {i+1}/{len(metadata)} from {entry['source']}")
            embedding = entry["embedding"]
            
            # Show embedding information
            logger.info(f"Embedding shape: {embedding.shape}, dtype: {embedding.dtype}")
            
            # Try each prompt
            for prompt_name, prompt in test_prompts.items():
                prompt_start = time.time()
                try:
                    logger.info(f"Testing prompt: '{prompt}'")
                    
                    # Set a per-prompt timeout
                    prompt_timeout = min(timeout / 2, 60)  # No more than 60 seconds per prompt
                    
                    # Check if we're close to the timeout
                    if time.time() - start_time > timeout - prompt_timeout:
                        logger.warning(f"Not enough time left to process prompt. Skipping.")
                        continue
                        
                    result = processor.infer_from_embedding(embedding, prompt)
                    prompt_time = time.time() - prompt_start
                    logger.info(f"Result for {prompt_name} (took {prompt_time:.2f}s): {result}")
                    entry[f"llava_{prompt_name}"] = result
                    success_count += 1
                except Exception as e:
                    logger.error(f"Error with prompt '{prompt_name}': {e}")
                    logger.error(traceback.format_exc())
                    entry[f"llava_{prompt_name}"] = f"Error: {str(e)}"
        
        # Check if at least some were successful
        if success_count > 0:
            logger.info(f"Successfully processed {success_count} prompts in {time.time() - start_time:.2f} seconds")
            return True
        else:
            logger.error("All embedding processing attempts failed")
            return False
            
    except Exception as e:
        logger.error(f"Error in test_llava_with_embeddings: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description="Test LLaVA with pre-computed embeddings")
    parser.add_argument("--embeddings", type=str, default="comparison_results/embeddings",
                        help="Directory containing embedding files (.pt or .pickle)")
    parser.add_argument("--model", type=str, default="llava-hf/llava-1.5-7b-hf",
                        help="LLaVA model name")
    parser.add_argument("--max-samples", type=int, default=2,
                        help="Maximum number of samples to process for testing")
    parser.add_argument("--timeout", type=int, default=180,
                        help="Maximum time in seconds to run the test")
    args = parser.parse_args()
    
    # Try each model in turn if the first one fails
    success = False
    
    if args.model != FAST_MODELS[0]:
        # Try the user-specified model
        logger.info(f"Testing with user-specified model: {args.model}")
        success = test_llava_with_embeddings(args.embeddings, args.model, args.max_samples, args.timeout)
        
    if not success:
        # Try our known working models
        for model in FAST_MODELS:
            if model == args.model:
                continue  # Skip if we already tried this
                
            logger.info(f"Trying alternative model: {model}")
            success = test_llava_with_embeddings(args.embeddings, model, args.max_samples, args.timeout)
            if success:
                break
    
    if success:
        logger.info("LLaVA with embeddings test completed successfully!")
        return 0
    else:
        logger.error("LLaVA with embeddings test failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 