"""
Mock LLaVA implementation for testing with CLIP embeddings.
This provides a lightweight testing system without requiring the full LLaVA model.
"""
import os
import torch
import logging
import argparse
import json
import pickle
import glob
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Prompt templates - same as in real implementation
PROMPT_TEMPLATES = {
    "caption": "Describe this image.",
    "scene": "What is happening in this image? Describe the scene in detail.",
    "qa": "What is the main object in this image? What is unusual or interesting here?",
    "explanation": "Explain the key elements and relationships in this image."
}

class MockLLaVAProcessor:
    """Mock LLaVA processor that generates plausible captions from embeddings."""
    
    def __init__(self, model_name: str = "mock-llava"):
        """Initialize mock LLaVA processor."""
        self.model_name = model_name
        logger.info(f"Initialized mock LLaVA processor with model: {model_name}")
        
        # We'll use different response styles based on the embedding values
        self.response_templates = {
            "nature": [
                "This is a beautiful natural landscape with mountains and trees.",
                "A serene nature scene with lush greenery and flowing water.",
                "An outdoor wilderness area with diverse plant life.",
                "A stunning view of natural terrain with various geological features."
            ],
            "urban": [
                "This appears to be a cityscape with tall buildings and urban infrastructure.",
                "An urban environment with streets, buildings, and city elements.",
                "A metropolitan area with architectural structures and civic design.",
                "A city scene showing various buildings and urban development."
            ],
            "person": [
                "The image shows a person engaged in some activity.",
                "A human figure appears in this image, possibly performing an action.",
                "There is a person visible in the scene, with certain distinguishing features.",
                "The image contains a human subject in a particular setting."
            ],
            "object": [
                "The image depicts a common object or item in focus.",
                "This seems to be a close-up view of an object or item.",
                "The main subject is an object with specific characteristics.",
                "A clear view of what appears to be an everyday item or object."
            ],
            "abstract": [
                "This appears to be an abstract image with various patterns and colors.",
                "The image contains abstract elements without a clear representational subject.",
                "A non-representational image with various visual elements and composition.",
                "An abstract composition featuring diverse visual properties and relationships."
            ]
        }
    
    def infer_from_embedding(self, embedding: torch.Tensor, prompt: str) -> str:
        """Generate mock caption from embedding vector."""
        logger.info(f"Generating mock response for prompt: '{prompt[:30]}...'")
        
        # Ensure embedding is a tensor
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding)
        
        # Use embedding statistics to determine response type
        embedding_mean = embedding.float().mean().item()
        embedding_std = embedding.float().std().item()
        
        # Determine category based on embedding statistics
        category = "abstract"  # default
        
        if embedding_std > 0.3:
            if embedding_mean > 0.1:
                category = "nature"
            else:
                category = "urban"
        else:
            if embedding_mean > 0:
                category = "person"
            else:
                category = "object"
        
        # Get a response from the appropriate category
        import random
        response_list = self.response_templates[category]
        base_response = random.choice(response_list)
        
        # Add more detail based on the prompt type
        if "describe" in prompt.lower() or "caption" in prompt.lower():
            detail = f" The {category} elements are prominently featured and create a compelling visual narrative."
        elif "main" in prompt.lower() or "subject" in prompt.lower():
            detail = f" The main subject appears to be the {category} elements which dominate the composition."
        elif "happening" in prompt.lower() or "scene" in prompt.lower():
            detail = f" The scene captures a moment in this {category} setting, conveying a sense of atmosphere and context."
        else:
            detail = f" Additional analysis shows interesting patterns typical of {category} imagery."
        
        # Return combined response
        full_response = base_response + detail
        logger.info(f"Generated mock response: {full_response[:50]}...")
        
        return full_response
    
    def batch_process_embeddings(self, metadata: List[Dict[str, Any]], 
                                 prompts: Dict[str, str] = None,
                                 embedding_key: str = "embedding") -> List[Dict[str, Any]]:
        """Process batch of embeddings with mock responses."""
        if prompts is None:
            prompts = {
                "llava_caption": PROMPT_TEMPLATES["caption"],
                "llava_qa": PROMPT_TEMPLATES["qa"],
            }
            
        for entry in tqdm(metadata, desc="Processing embeddings"):
            if embedding_key not in entry:
                logger.warning(f"No embedding found in entry")
                continue
                
            embedding = entry[embedding_key]
            
            for key, prompt in prompts.items():
                try:
                    entry[key] = self.infer_from_embedding(embedding, prompt)
                except Exception as e:
                    error_msg = f"Error processing with prompt '{key}': {str(e)}"
                    logger.error(error_msg)
                    entry[key] = f"Error: {str(e)}"
                    
        return metadata

def load_embeddings(embeddings_dir, limit=10):
    """Load embeddings from directory."""
    logger.info(f"Loading embeddings from: {embeddings_dir}")
    
    embeddings = []
    paths = []
    
    # Try to find PT files
    pt_files = sorted(glob.glob(os.path.join(embeddings_dir, "*.pt")))
    if pt_files:
        logger.info(f"Found {len(pt_files)} PT files")
        file_count = min(len(pt_files), limit)
        
        # Filter out metadata files
        pt_files = [f for f in pt_files if "metadata" not in f.lower()]
        
        for pt_file in pt_files[:file_count]:
            try:
                # Try to load with weights_only=True
                embedding = torch.load(pt_file, weights_only=True, map_location="cpu")
                
                # Handle various embedding formats
                if isinstance(embedding, dict):
                    # Try to extract tensor from dict
                    for key, value in embedding.items():
                        if isinstance(value, torch.Tensor) and len(value.shape) > 0:
                            embeddings.append(value)
                            paths.append(f"{pt_file}:{key}")
                            break
                elif isinstance(embedding, torch.Tensor):
                    embeddings.append(embedding)
                    paths.append(pt_file)
            except Exception as e:
                logger.error(f"Error loading {pt_file}: {e}")
                # Try alternative loading method
                try:
                    embedding = torch.load(pt_file, weights_only=False, map_location="cpu")
                    if isinstance(embedding, torch.Tensor):
                        embeddings.append(embedding)
                        paths.append(pt_file)
                except Exception as inner_e:
                    logger.error(f"Still could not load {pt_file}: {inner_e}")
    
    # If no embeddings loaded, create random ones for testing
    if not embeddings:
        logger.warning("No embeddings loaded, creating random ones for testing")
        for i in range(3):
            embeddings.append(torch.randn(512))  # Standard CLIP size
            paths.append(f"random_embedding_{i}")
    
    # Create metadata list
    metadata = []
    for i, (emb, path) in enumerate(zip(embeddings, paths)):
        if not isinstance(emb, torch.Tensor):
            continue
            
        metadata.append({
            "id": i,
            "embedding": emb,
            "source": path
        })
    
    logger.info(f"Successfully loaded {len(metadata)} embeddings")
    return metadata

def main():
    parser = argparse.ArgumentParser(description="Mock LLaVA processor for testing with CLIP embeddings")
    parser.add_argument("--embeddings", type=str, default="comparison_results/embeddings",
                        help="Directory containing embedding files")
    parser.add_argument("--output", type=str, default="mock_llava_results.json",
                        help="Output file for results")
    parser.add_argument("--limit", type=int, default=10,
                        help="Maximum number of embeddings to process")
    args = parser.parse_args()
    
    # Load embeddings
    metadata = load_embeddings(args.embeddings, args.limit)
    if not metadata:
        logger.error("No embeddings could be loaded")
        return 1
    
    # Create mock processor
    processor = MockLLaVAProcessor()
    
    # Process embeddings
    logger.info("Processing embeddings with mock LLaVA")
    results = processor.batch_process_embeddings(metadata)
    
    # Save results
    logger.info(f"Saving results to {args.output}")
    with open(args.output, 'w') as f:
        # Convert tensor objects to lists for JSON serialization
        for entry in results:
            if "embedding" in entry and isinstance(entry["embedding"], torch.Tensor):
                entry["embedding"] = entry["embedding"].tolist()
        
        json.dump(results, f, indent=2)
    
    logger.info(f"Successfully processed {len(results)} embeddings with mock LLaVA")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main()) 