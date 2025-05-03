#!/usr/bin/env python3
"""
Test for SigLIP model using Flickr30K dataset.
Purpose: Evaluates SigLIP on real-world Flickr30K dataset
Features:
Uses the custom model wrapper
Implements batch processing
Calculates Recall@K metrics for both image-to-text and text-to-image
Includes visualization functionality
Has command-line arguments for customization
Contains proper logging
Loads real image-caption data from the Flickr30K dataset
"""

import os
import sys
import time
import torch
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the SigLIP model
from model.siglip_model import load_siglip_model

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_siglip_with_flickr30k(num_samples=100, batch_size=16):
    """
    Test SigLIP model with Flickr30K dataset.
    
    Args:
        num_samples: Number of samples to test (default: 100)
        batch_size: Batch size for processing (default: 16)
    """
    # Step 1: Load the SigLIP model
    logger.info("Loading SigLIP model...")
    model = load_siglip_model(
        model_name="ViT-B-16-SigLIP",
        pretrained="webli",
        device=None,  # Auto-select device
        initial_batch_size=4,
        max_batch_size=batch_size,
        cache_embeddings=True
    )
    logger.info(f"SigLIP model loaded successfully: {model.model_name} with {model.pretrained} weights on {model.device}")

    # Step 2: Load Flickr30K dataset
    logger.info("Loading Flickr30K dataset...")
    try:
        dataset = load_dataset("nlphuji/flickr30k", split="test")
        logger.info(f"Dataset loaded successfully. Columns: {dataset.column_names}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return False

    # Step 3: Prepare samples for testing
    logger.info(f"Preparing {num_samples} samples for testing...")
    samples = []
    
    # Ensure we don't request more samples than available
    num_samples = min(num_samples, len(dataset))
    
    # Get a subset of samples
    for i in range(num_samples):
        try:
            sample = dataset[i]
            # Flickr30K provides 5 captions per image
            samples.append({
                "image": sample["image"],
                "captions": sample["caption"]
            })
        except Exception as e:
            logger.error(f"Error processing sample {i}: {e}")
            continue
    
    logger.info(f"Successfully prepared {len(samples)} samples")

    # Step 4: Process images and texts in batches
    logger.info("Processing images and texts...")
    
    # Prepare batches
    image_batches = []
    caption_batches = []
    
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        
        # Create image batch
        image_batch = [sample["image"] for sample in batch]
        image_batches.append(image_batch)
        
        # Use just the first caption for each image
        caption_batch = [sample["captions"][0] for sample in batch]
        caption_batches.append(caption_batch)
    
    # Process image batches
    logger.info("Encoding images...")
    all_image_embeddings = []
    
    start_time = time.time()
    for batch in tqdm(image_batches):
        result = model.encode_images(batch)
        all_image_embeddings.append(result["image_embeddings"])
    
    image_encoding_time = time.time() - start_time
    logger.info(f"Image encoding completed in {image_encoding_time:.2f} seconds")
    
    # Process text batches
    logger.info("Encoding texts...")
    all_text_embeddings = []
    
    start_time = time.time()
    for batch in tqdm(caption_batches):
        result = model.encode_text(batch)
        all_text_embeddings.append(result["text_embeddings"])
    
    text_encoding_time = time.time() - start_time
    logger.info(f"Text encoding completed in {text_encoding_time:.2f} seconds")
    
    # Step 5: Compute similarity between all images and texts
    logger.info("Computing similarity matrix...")
    
    # Concatenate all embeddings
    image_embeddings = torch.cat(all_image_embeddings, dim=0)
    text_embeddings = torch.cat(all_text_embeddings, dim=0)
    
    # Compute similarity matrix
    similarity = model.compute_similarity(image_embeddings, text_embeddings)
    
    # Step 6: Evaluate retrieval performance
    logger.info("Evaluating retrieval performance...")
    
    # Compute metrics - Recall@K
    r_at_1 = recall_at_k(similarity, k=1)
    r_at_5 = recall_at_k(similarity, k=5)
    r_at_10 = recall_at_k(similarity, k=10)
    
    logger.info(f"Image-to-Text Retrieval Results:")
    logger.info(f"Recall@1: {r_at_1:.4f}")
    logger.info(f"Recall@5: {r_at_5:.4f}")
    logger.info(f"Recall@10: {r_at_10:.4f}")
    
    # Also evaluate text-to-image retrieval
    similarity_t2i = similarity.t()  # Transpose for text-to-image
    
    r_at_1_t2i = recall_at_k(similarity_t2i, k=1)
    r_at_5_t2i = recall_at_k(similarity_t2i, k=5)
    r_at_10_t2i = recall_at_k(similarity_t2i, k=10)
    
    logger.info(f"Text-to-Image Retrieval Results:")
    logger.info(f"Recall@1: {r_at_1_t2i:.4f}")
    logger.info(f"Recall@5: {r_at_5_t2i:.4f}")
    logger.info(f"Recall@10: {r_at_10_t2i:.4f}")
    
    # Step 7: Visualize a few examples
    try:
        visualize_examples(samples, similarity, num_examples=3)
    except Exception as e:
        logger.error(f"Error visualizing examples: {e}")
    
    logger.info("SigLIP test with Flickr30K completed successfully!")
    return True

def recall_at_k(similarity, k=1):
    """
    Compute Recall@K metric.
    
    Args:
        similarity: Similarity matrix (image x text)
        k: K value for recall
    
    Returns:
        Recall@K value
    """
    # Get the indices of the K highest similarities per row (image)
    _, indices = similarity.topk(k=k, dim=1)
    
    # Generate diagonal indices (ground truth matches)
    diagonal = torch.arange(similarity.size(0))
    
    # Check if the ground truth index is in the top K predictions
    correct = torch.any(indices == diagonal.unsqueeze(-1).expand_as(indices), dim=1)
    
    # Calculate recall@K
    recall = torch.mean(correct.float()).item()
    return recall

def visualize_examples(samples, similarity, num_examples=3, output_dir="result"):
    """
    Visualize a few examples of image-text retrieval.
    
    Args:
        samples: Dataset samples
        similarity: Similarity matrix
        num_examples: Number of examples to visualize
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get the indices of the highest similarities per row (image)
    _, indices = similarity.topk(k=1, dim=1)
    
    # Select a few examples
    for i in range(min(num_examples, len(samples))):
        # Get sample and predicted caption
        sample = samples[i]
        image = sample["image"]
        
        actual_caption = sample["captions"][0]
        predicted_idx = indices[i, 0].item()
        predicted_caption = samples[predicted_idx]["captions"][0]
        
        # Plot image and captions
        plt.figure(figsize=(10, 6))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Example {i+1}")
        
        plt.figtext(0.1, 0.1, f"Actual: {actual_caption}", wrap=True, fontsize=10)
        plt.figtext(0.1, 0.05, f"Predicted: {predicted_caption}", wrap=True, fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/example_{i+1}.png")
        plt.close()
    
    logger.info(f"Visualizations saved to {output_dir} directory")

if __name__ == "__main__":
    logger.info("Starting SigLIP test with Flickr30K")
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SigLIP with Flickr30K dataset.")
    parser.add_argument("--num_samples", type=int, default=100, 
                        help="Number of samples to test (default: 100)")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for processing (default: 16)")
    
    args = parser.parse_args()
    
    test_siglip_with_flickr30k(
        num_samples=args.num_samples,
        batch_size=args.batch_size
    ) 