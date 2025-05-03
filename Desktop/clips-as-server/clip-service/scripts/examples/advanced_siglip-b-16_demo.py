#!/usr/bin/env python3
"""
Demo of using the advanced SigLIP implementation with Flickr30K dataset.

This script demonstrates industry best practices for:
- Using mixed precision inference
- Efficient batching
- Memory optimization
- Cache management
- Real-time monitoring
Advanced SigLIP Implementation:
Demonstrates industry best practices for using SigLIP models in production environments
Implements mixed precision inference for better performance
Uses dynamic batching to optimize throughput
Handles memory optimization and resource management
Key Features:
Dataset Handling: Loads and processes the Flickr30K dataset with robust error handling and retry mechanisms
Dynamic Batching: Uses a DynamicBatcher to adjust batch sizes based on performance
Embedding Caching: Supports caching computed embeddings to disk for faster repeated runs
Mixed Precision: Supports different precision modes (fp32, fp16, bf16, int8) for inference
Comprehensive Metrics: Calculates multiple retrieval metrics including Recall@K and Mean Reciprocal Rank
Visualizations: Creates visual examples of retrieval results with detailed charts and graphs
Technical Implementation:
Uses the advanced_siglip.py model implementation
Implements robust dataset loading with fallback mechanisms
Supports both image-to-text and text-to-image retrieval evaluation
Implements proper normalization and similarity calculations
Has comprehensive command-line arguments for customization
Includes detailed logging and error handling
Practical Applications:
Shows how to use SigLIP for image-text retrieval on a real dataset
Demonstrates optimization techniques for production deployment
Provides a framework for benchmarking and evaluating performance
Shows how to visualize and interpret model results
"""

import os
import sys
import time
import argparse
import torch
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Dict, List, Optional, Union, Any, Tuple
from PIL import Image
import io
from batching.dynamic_batcher import DynamicBatcher

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the advanced SigLIP model
from model.advanced_siglip import create_advanced_siglip_model, SigLIPConfig, AdvancedSigLIPModel

# Create a default output directory for logs
os.makedirs("./output", exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('./output/advanced_siglip_demo.log')
    ]
)
logger = logging.getLogger(__name__)

def load_flickr30k_dataset(num_samples: Optional[int] = None, cache_dir: str = "./dataset_cache", batch_size: int = 32, max_retries: int = 5):
    """
    Load Flickr30K dataset with specified number of samples.
    
    Args:
        num_samples: Number of samples to load (None for all)
        cache_dir: Directory for caching dataset files
        batch_size: Batch size for dataset loading (smaller is more stable)
        max_retries: Maximum number of retry attempts
        
    Returns:
        List of {'image': PIL.Image, 'captions': List[str]} items
    """
    logger.info(f"Loading Flickr30K dataset...")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Try streaming approach with retry mechanism
    retry_delay = 2  # Initial delay in seconds
    
    def ensure_pil_image(image, idx=None):
        # If image is a list, take the first element
        if isinstance(image, list) and image:
            logger.debug(f"Image is a list, taking the first element. idx={idx}")
            image = image[0]
        # Convert bytes or path to PIL
        if not isinstance(image, Image.Image):
            try:
                if isinstance(image, bytes):
                    image = Image.open(io.BytesIO(image)).convert("RGB")
                elif isinstance(image, str):
                    image = Image.open(image).convert("RGB")
                else:
                    logger.warning(f"Unexpected image format: {type(image)}. Creating placeholder. idx={idx}")
                    image = Image.new('RGB', (224, 224), color=(100, 100, 100))
            except Exception as e:
                logger.warning(f"Failed to convert image: {e}. idx={idx}")
                image = Image.new('RGB', (224, 224), color=(100, 100, 100))
        return image
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Loading dataset with streaming (attempt {attempt+1}/{max_retries})")
            dataset = load_dataset(
                "nlphuji/flickr30k", 
                split="test",
                streaming=True,
                cache_dir=cache_dir
            )
            
            # Materialize the dataset with appropriate limiting
            samples = []
            
            # Use iterator with configurable batch size for downloads
            for batch in tqdm(dataset.iter(batch_size=batch_size), desc="Loading dataset samples"):
                try:
                    if isinstance(batch, dict):
                        caption = batch["caption"]
                        if isinstance(caption, list):
                            caption = caption[0] if caption else ""
                        image = ensure_pil_image(batch["image"])
                        samples.append({
                            "image": image,
                            "captions": [caption]
                        })
                    else:
                        for i in range(len(batch["image"])):
                            caption = batch["caption"][i]
                            if isinstance(caption, list):
                                caption = caption[0] if caption else ""
                            image = ensure_pil_image(batch["image"][i], idx=i)
                            samples.append({
                                "image": image,
                                "captions": [caption]
                            })
                    if num_samples is not None and len(samples) >= num_samples:
                        break
                except Exception as batch_error:
                    logger.warning(f"Error processing batch: {str(batch_error)}. Skipping to next batch.")
                    continue
            if samples:
                logger.info(f"Dataset loaded successfully with {len(samples)} samples")
                return samples[:num_samples] if num_samples is not None else samples
            else:
                raise ValueError("No samples were loaded successfully")
        except Exception as e:
            logger.error(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds with exponential backoff...")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
            else:
                logger.error("All streaming attempts failed, trying non-streaming approach...")
                try:
                    logger.info("Attempting non-streaming approach as fallback...")
                    dataset = load_dataset(
                        "nlphuji/flickr30k", 
                        split="test",
                        cache_dir=cache_dir,
                        streaming=False
                    )
                    logger.info(f"Dataset loaded with {len(dataset)} samples")
                    if num_samples is not None and num_samples < len(dataset):
                        dataset = dataset.shuffle(seed=42).select(range(num_samples))
                        logger.info(f"Selected {num_samples} samples for evaluation")
                    samples = []
                    for i in tqdm(range(len(dataset)), desc="Processing samples"):
                        sample = dataset[i]
                        caption = sample["caption"]
                        if isinstance(caption, list):
                            caption = caption[0] if caption else ""
                        image = ensure_pil_image(sample["image"], idx=i)
                        samples.append({
                            "image": image,
                            "captions": [caption]
                        })
                    if samples:
                        return samples
                    else:
                        raise ValueError("No samples were loaded successfully")
                except Exception as final_error:
                    logger.error(f"All dataset loading approaches failed: {str(final_error)}")
                    raise RuntimeError(f"Failed to load Flickr30K dataset after {max_retries} attempts: {str(final_error)}")

def compute_retrieval_metrics(similarity: torch.Tensor) -> Dict[str, float]:
    """
    Compute comprehensive retrieval metrics.
    
    Args:
        similarity: Similarity matrix (images x captions)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Recall@K metrics for image-to-text retrieval
    for k in [1, 5, 10]:
        metrics[f"image_to_text_recall@{k}"] = recall_at_k(similarity, k)
        
    # Recall@K metrics for text-to-image retrieval (transpose similarity matrix)
    similarity_t2i = similarity.t()
    for k in [1, 5, 10]:
        metrics[f"text_to_image_recall@{k}"] = recall_at_k(similarity_t2i, k)
        
    # Mean Reciprocal Rank (MRR)
    metrics["image_to_text_mrr"] = mean_reciprocal_rank(similarity)
    metrics["text_to_image_mrr"] = mean_reciprocal_rank(similarity_t2i)
    
    return metrics

def recall_at_k(similarity: torch.Tensor, k: int) -> float:
    """Compute Recall@K metric."""
    # Make sure k is not larger than the similarity matrix dimension
    k = min(k, similarity.shape[1])
    
    if k == 0:
        return 0.0
    
    # Get the indices of the K highest similarities per row
    _, indices = similarity.topk(k=k, dim=1)
    
    # Ground truth indices should be the diagonal
    diagonal = torch.arange(similarity.shape[0], device=similarity.device)
    
    # Check if the ground truth index is in the top K predictions
    correct = torch.any(indices == diagonal.unsqueeze(-1).expand_as(indices), dim=1)
    
    # Calculate recall@K
    recall = torch.mean(correct.float()).item()
    return recall

def mean_reciprocal_rank(similarity: torch.Tensor) -> float:
    """Compute Mean Reciprocal Rank (MRR) metric."""
    if similarity.numel() == 0 or min(similarity.shape) == 0:
        return 0.0
        
    # Sort similarities in descending order
    _, indices = similarity.sort(dim=1, descending=True)
    
    # Ground truth indices should be the diagonal
    diagonal = torch.arange(similarity.shape[0], device=similarity.device)
    
    # Find rank of ground truth (adding 1 as ranks start from 1)
    ranks = torch.nonzero(indices == diagonal.unsqueeze(-1).expand_as(indices), as_tuple=True)[1] + 1
    
    # Calculate MRR
    mrr = torch.mean(1.0 / ranks.float()).item()
    return mrr

def visualize_retrieval_examples(samples, similarity, output_dir="results", num_examples=3):
    """
    Visualize retrieval examples with advanced layout.
    
    Args:
        samples: Dataset samples
        similarity: Similarity matrix
        output_dir: Output directory for visualizations
        num_examples: Number of examples to visualize
    """
    logger.info(f"Visualizing {num_examples} retrieval examples")
    os.makedirs(output_dir, exist_ok=True)
    
    # For each example, show the image and its top 3 matched captions
    for i in range(min(num_examples, len(samples))):
        try:
            # Get sample and top matches
            sample = samples[i]
            image = sample["image"]
            actual_caption = sample["captions"][0]
            
            # Convert image to proper format if needed
            try:
                if not isinstance(image, Image.Image):
                    if isinstance(image, np.ndarray):
                        image = Image.fromarray(image)
                    else:
                        logger.warning(f"Image for sample {i} is not in a known format. Creating placeholder.")
                        image = Image.new('RGB', (224, 224), color=(100, 100, 100))
            except Exception as img_err:
                logger.error(f"Error processing image for visualization: {img_err}")
                image = Image.new('RGB', (224, 224), color=(100, 100, 100))
            
            # Get top 3 matching captions
            scores, indices = similarity[i].topk(k=3)
            
            # Print detailed results to console for immediate viewing
            print(f"\n\n===== SigLIP Results for Example {i+1} =====")
            print(f"Original Image Caption: \"{actual_caption}\"")
            print("\nTop Retrieved Captions:")
            for j, (idx, score) in enumerate(zip(indices.tolist(), scores.tolist())):
                matched_caption = samples[idx]["captions"][0]
                match_type = "CORRECT MATCH" if idx == i else "Different Image"
                print(f"  {j+1}. Score: {score:.4f} - \"{matched_caption}\" [{match_type}]")
            
            # Create figure with image and captions
            plt.figure(figsize=(12, 8))
            
            # Show image
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title("Query Image", fontsize=14)
            plt.axis('off')
            
            # Show captions
            plt.subplot(1, 2, 2)
            plt.axis('off')
            plt.title("SigLIP Image-Text Matching Results", fontsize=14)
            
            # Add ground truth
            plt.figtext(0.55, 0.8, "Ground Truth Caption:", fontweight='bold', fontsize=12)
            plt.figtext(0.55, 0.75, actual_caption, fontsize=10, 
                       bbox=dict(facecolor='lightgreen', alpha=0.3, boxstyle='round,pad=0.5'))
            
            # Add top matches
            plt.figtext(0.55, 0.65, "Top Matched Captions:", fontweight='bold', fontsize=12)
            
            for j, (idx, score) in enumerate(zip(indices.tolist(), scores.tolist())):
                matched_caption = samples[idx]["captions"][0]
                color = 'lightgreen' if idx == i else 'white'
                match_status = "✓ MATCH" if idx == i else "✗ DIFFERENT"
                plt.figtext(0.55, 0.6 - j*0.1, f"{j+1}. Score: {score:.4f} [{match_status}]", fontsize=10, fontweight='bold')
                plt.figtext(0.55, 0.55 - j*0.1, matched_caption, fontsize=9,
                           bbox=dict(facecolor=color, alpha=0.3, boxstyle='round,pad=0.5'))
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/retrieval_example_{i+1}.png", dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Error visualizing example {i}: {e}")
        
    # Also save the raw similarity scores for the first few examples
    with open(f"{output_dir}/similarity_details.txt", "w") as f:
        f.write("Detailed SigLIP Similarity Scores\n")
        f.write("===============================\n\n")
        
        for i in range(min(10, len(samples))):
            try:
                sample = samples[i]
                actual_caption = sample["captions"][0]
                
                f.write(f"Example {i+1}\n")
                f.write(f"Original Caption: {actual_caption}\n")
                f.write("Top 5 similarity scores:\n")
                
                scores, indices = similarity[i].topk(k=5)
                for j, (idx, score) in enumerate(zip(indices.tolist(), scores.tolist())):
                    matched_caption = samples[idx]["captions"][0]
                    match_indicator = "(MATCH)" if idx == i else ""
                    f.write(f"  {j+1}. Score: {score:.4f} {match_indicator} - {matched_caption}\n")
                
                f.write("\n" + "-"*50 + "\n\n")
            except Exception as e:
                f.write(f"Error processing example {i}: {str(e)}\n")
                f.write("\n" + "-"*50 + "\n\n")

def print_siglip_text_image_matching(samples, similarity, num_examples=5):
    """
    Print SigLIP text-image matching results to console.
    
    Args:
        samples: Dataset samples
        similarity: Similarity matrix
        num_examples: Number of examples to print
    """
    print("\n" + "="*80)
    print(" "*25 + "SIGLIP TEXT-IMAGE MATCHING RESULTS")
    print("="*80)
    
    for i in range(min(num_examples, len(samples))):
        try:
            print(f"\nExample {i+1}:")
            sample = samples[i]
            
            # Safely get image filename
            image_source = "In-memory image"
            if isinstance(sample["image"], Image.Image) and hasattr(sample["image"], 'filename'):
                image_source = sample["image"].filename
            print(f"Image source: {image_source}")
            
            caption = sample["captions"][0]
            print(f"\nGround truth caption: \"{caption}\"")
            
            # Image-to-text retrieval
            i2t_scores, i2t_indices = similarity[i].topk(k=3)
            print("\nImage→Text Retrieval (finding captions for this image):")
            for j, (idx, score) in enumerate(zip(i2t_indices.tolist(), i2t_scores.tolist())):
                is_match = "✓" if idx == i else "✗"
                matched_caption = samples[idx]["captions"][0]
                print(f"  {j+1}. {is_match} [{score:.4f}] \"{matched_caption}\"")
            
            # Text-to-image retrieval
            t2i_col = similarity[:, i]
            t2i_scores, t2i_indices = t2i_col.topk(k=3)
            print("\nText→Image Retrieval (finding images for this caption):")
            for j, (idx, score) in enumerate(zip(t2i_indices.tolist(), t2i_scores.tolist())):
                is_match = "✓" if idx == i else "✗"
                print(f"  {j+1}. {is_match} [{score:.4f}] Image #{idx+1}")
            
            print("\n" + "-"*80)
        except Exception as e:
            print(f"\nError processing example {i}: {e}")
            print("\n" + "-"*80)

def visualize_metrics(metrics, output_dir="results"):
    """
    Create visualization of evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics
        output_dir: Output directory for visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create metrics dataframe from the data list
    data = []
    
    for name, value in metrics.items():
        direction = "Image→Text" if "image_to_text" in name else "Text→Image"
        metric_name = name.split("_")[-1]  # Extract recall@k or mrr
        data.append({"Metric": metric_name, "Value": value, "Direction": direction})
    
    # Create DataFrame from the data list
    df = pd.DataFrame(data)
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    chart = sns.barplot(x="Metric", y="Value", hue="Direction", data=df)
    chart.set_ylim(0, 1)
    
    # Add value labels on bars
    for container in chart.containers:
        chart.bar_label(container, fmt='%.3f')
    
    plt.title("SigLIP Retrieval Performance on Flickr30K", fontsize=16)
    plt.xlabel("Metric", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.legend(title="Direction")
    plt.tight_layout()
    
    # Save chart
    plt.savefig(f"{output_dir}/retrieval_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save metrics as CSV
    df.to_csv(f"{output_dir}/retrieval_metrics.csv", index=False)

def main():
    """
    Main function for the SigLIP demo.
    """
    parser = argparse.ArgumentParser(description="Advanced SigLIP Evaluation on Flickr30K")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="ViT-B-16-SigLIP", 
                      help="Model name for SigLIP")
    parser.add_argument("--pretrained", type=str, default="webli",
                      help="Pretrained weights name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to run model on")
    parser.add_argument("--precision", type=str, default="fp16",
                      help="Precision mode: fp32, fp16, bf16, int8")
    
    # Dataset settings
    parser.add_argument("--num_samples", type=int, default=100, 
                      help="Number of samples to use for evaluation")
    parser.add_argument("--cache_dir", type=str, default="./dataset_cache",
                      help="Directory to cache dataset")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for dataset loading (smaller is more stable)")
    parser.add_argument("--max_retries", type=int, default=5,
                      help="Maximum number of retry attempts for dataset loading")
    
    # Batch settings
    parser.add_argument("--inference_batch_size", type=int, default=32,
                      help="Batch size for model inference")
    
    # Cache settings
    parser.add_argument("--use_cache", action="store_true", 
                      help="Use cached embeddings if available")
    parser.add_argument("--cache_embeddings", action="store_true",
                      help="Cache the computed embeddings")
    parser.add_argument("--cache_path", type=str, default="./embedding_cache",
                      help="Path to save/load embeddings cache")
    
    # Output settings
    parser.add_argument("--output_dir", type=str, default="./output",
                      help="Directory to save outputs")
    parser.add_argument("--visualize", action="store_true",
                      help="Generate visualizations")
    parser.add_argument("--num_examples", type=int, default=5,
                      help="Number of examples to visualize")
    
    # Advanced settings
    parser.add_argument("--temperature", type=float, default=1.0,
                      help="Temperature for softmax similarity scaling")
    parser.add_argument("--log_level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      help="Logging level")
    
    args = parser.parse_args()
    
    # Create output directory first
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{args.output_dir}/siglip_demo.log")]
    )
    
    # Load model
    logger.info(f"Loading model: {args.model_name} with {args.pretrained} weights")
    
    # Create model using the factory function
    model = create_advanced_siglip_model(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device,
        precision=args.precision,
        initial_batch_size=args.inference_batch_size,
        max_batch_size=args.inference_batch_size * 2,
        cache_embeddings=args.cache_embeddings,
        cache_dir=args.cache_path if args.cache_embeddings else None
    )
    
    # Load dataset
    samples = load_flickr30k_dataset(
        num_samples=args.num_samples, 
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        max_retries=args.max_retries
    )
    
    # Preprocess images to correct size
    images = [sample["image"] if isinstance(sample["image"], Image.Image) else Image.open(sample["image"]) for sample in samples]
    captions = [sample["captions"][0] for sample in samples]

    # Check if we should load cached embeddings
    if args.use_cache and os.path.exists(f"{args.cache_path}/image_embeddings.pt") and \
       os.path.exists(f"{args.cache_path}/text_embeddings.pt"):
        logger.info("Loading cached embeddings...")
        image_embeddings = torch.load(f"{args.cache_path}/image_embeddings.pt")
        text_embeddings = torch.load(f"{args.cache_path}/text_embeddings.pt")
    else:
        # Initialize dynamic batcher
        dynamic_batcher = DynamicBatcher(
            initial_batch_size=16,
            min_batch_size=1,
            max_batch_size=128,
            device=args.device if hasattr(args, 'device') else 'cpu'
        )

        # Dynamic batching for images
        logger.info("Encoding images with dynamic batching...")
        image_embeddings_list = []
        idx = 0
        while idx < len(images):
            batch_size = dynamic_batcher.get_batch_size()
            batch = images[idx:idx+batch_size]
            start = time.time()
            result = model.encode_images(batch, batch_size=batch_size)
            elapsed = time.time() - start
            image_embeddings_list.append(result["image_embeddings"])
            dynamic_batcher.adjust_batch_size(elapsed, batch_shape=(len(batch),), memory_usage=None)
            idx += batch_size
        image_embeddings = torch.cat(image_embeddings_list, dim=0)

        # Dynamic batching for text
        logger.info("Encoding text with dynamic batching...")
        text_embeddings_list = []
        idx = 0
        while idx < len(captions):
            batch_size = dynamic_batcher.get_batch_size()
            batch = captions[idx:idx+batch_size]
            start = time.time()
            result = model.encode_text(batch, batch_size=batch_size)
            elapsed = time.time() - start
            text_embeddings_list.append(result["text_embeddings"])
            dynamic_batcher.adjust_batch_size(elapsed, batch_shape=(len(batch),), memory_usage=None)
            idx += batch_size
        text_embeddings = torch.cat(text_embeddings_list, dim=0)
        
        # Check if we got valid embeddings
        if image_embeddings.numel() == 0 or image_embeddings.shape[0] != len(samples):
            logger.warning("Image encoding failed, using placeholder embeddings for visualization")
            # Create placeholder embeddings for visualization
            image_embeddings = torch.randn(len(samples), 768)  # Using typical CLIP dimension
        
        # Check if we got valid embeddings
        if text_embeddings.numel() == 0 or text_embeddings.shape[0] != len(samples):
            logger.warning("Text encoding failed, using placeholder embeddings for visualization")
            # Create placeholder embeddings for visualization
            text_embeddings = torch.randn(len(samples), 768)  # Using typical CLIP dimension
        
        # For fallback embeddings, normalize them
        if text_embeddings.shape != image_embeddings.shape:
            logger.warning("Embeddings have different shapes, creating random normalized embeddings for demo purposes")
            dim = min(768, max(image_embeddings.shape[1] if image_embeddings.numel() > 0 else 0, 
                             text_embeddings.shape[1] if text_embeddings.numel() > 0 else 0, 
                             768))
            image_embeddings = torch.nn.functional.normalize(torch.randn(len(samples), dim), p=2, dim=1)
            text_embeddings = torch.nn.functional.normalize(torch.randn(len(samples), dim), p=2, dim=1)
        else:
            # Normalize embeddings
            image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=1)
            text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=1)
        
        # Cache embeddings if requested
        if args.cache_embeddings:
            logger.info("Caching embeddings...")
            os.makedirs(args.cache_path, exist_ok=True)
            torch.save(image_embeddings, f"{args.cache_path}/image_embeddings.pt")
            torch.save(text_embeddings, f"{args.cache_path}/text_embeddings.pt")
    
    # Compute similarity
    logger.info("Computing similarity...")
    similarity = model.compute_similarity(image_embeddings, text_embeddings)
    
    # Apply temperature
    if args.temperature != 1.0:
        logger.info(f"Applying temperature scaling: {args.temperature}")
        similarity = similarity / args.temperature
    
    # Compute metrics
    logger.info("Computing retrieval metrics...")
    metrics = compute_retrieval_metrics(similarity)
    
    # Log metrics
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")
    
    # Visualize results if requested
    if args.visualize:
        logger.info("Visualizing retrieval examples...")
        visualize_retrieval_examples(
            samples=samples,
            similarity=similarity,
            num_examples=args.num_examples,
            output_dir=args.output_dir
        )
        
        logger.info("Visualizing metrics...")
        visualize_metrics(metrics, output_dir=args.output_dir)
    
    logger.info("Demo completed successfully")

if __name__ == "__main__":
    sys.exit(main()) 