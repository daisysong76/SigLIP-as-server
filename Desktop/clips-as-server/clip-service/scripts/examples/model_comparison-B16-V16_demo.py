#!/usr/bin/env python3
"""
Comparative evaluation of SigLIP models (ViT-B vs ViT-L) on Flickr30K dataset.
This script demonstrates:
- Performance comparison between model sizes
- Reranking for improved retrieval accuracy
- Side-by-side comparison of results
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
from typing import Dict, List, Optional, Tuple
from PIL import Image
from contextlib import nullcontext

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the advanced SigLIP model
from model.advanced_siglip import create_advanced_siglip_model, SigLIPConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_comparison.log')
    ]
)
logger = logging.getLogger(__name__)

def load_flickr30k_dataset(num_samples: Optional[int] = None, cache_dir: str = "./dataset_cache"):
    """
    Load Flickr30K dataset with specified number of samples.
    
    Args:
        num_samples: Number of samples to load (None for all)
        cache_dir: Directory for caching dataset files
        
    Returns:
        List of {'image': PIL.Image, 'captions': List[str]} items
    """
    logger.info(f"Loading Flickr30K dataset...")
    
    # Try non-streaming approach first since it's more reliable
    try:
        dataset = load_dataset(
            "nlphuji/flickr30k", 
            split="test",
            cache_dir=cache_dir,
            streaming=False
        )
        
        logger.info(f"Dataset loaded with {len(dataset)} samples")
        
        # Select samples if specified
        if num_samples is not None and num_samples < len(dataset):
            # Use stratified sampling for more representative results
            dataset = dataset.shuffle(seed=42).select(range(num_samples))
            logger.info(f"Selected {num_samples} samples for evaluation")
        
        # Extract images and captions
        samples = []
        for i in tqdm(range(len(dataset)), desc="Processing samples"):
            sample = dataset[i]
            # Ensure caption is a string, not a list
            caption = sample["caption"]
            if isinstance(caption, list):
                caption = caption[0] if caption else ""
            
            samples.append({
                "image": sample["image"],  # This is already a PIL Image
                "captions": [caption]  # List with a single string caption
            })
        
        return samples
            
    except Exception as e:
        logger.error(f"Error loading dataset with non-streaming approach: {str(e)}")
        logger.info("Trying with streaming approach...")
        
        # Fallback to streaming with cache
        try:
            dataset = load_dataset(
                "nlphuji/flickr30k", 
                split="test", 
                streaming=True,
                cache_dir=cache_dir
            )
            
            # Materialize the dataset with appropriate limiting
            samples = []
            
            # Use iterator with small batch size for downloads
            for batch in tqdm(dataset.iter(batch_size=32), desc="Loading dataset samples"):
                # Process a single item
                if isinstance(batch, dict):
                    # Get the caption as a string
                    caption = batch["caption"]
                    if isinstance(caption, list):
                        caption = caption[0] if caption else ""
                    
                    samples.append({
                        "image": batch["image"],  # This should be a PIL Image
                        "captions": [caption]  # List with a single string caption
                    })
                else:
                    # Process a batch of items
                    for i in range(len(batch["image"])):
                        # Get the caption as a string
                        caption = batch["caption"][i]
                        if isinstance(caption, list):
                            caption = caption[0] if caption else ""
                        
                        samples.append({
                            "image": batch["image"][i],  # This should be a PIL Image
                            "captions": [caption]  # List with a single string caption
                        })
                
                # Check if we've reached our sample limit
                if num_samples is not None and len(samples) >= num_samples:
                    break
                
            logger.info(f"Dataset loaded with {len(samples)} samples")
            return samples[:num_samples] if num_samples is not None else samples
            
        except Exception as e:
            logger.error(f"Failed to load dataset after retry: {str(e)}")
            raise

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
    # Sort similarities in descending order
    _, indices = similarity.sort(dim=1, descending=True)
    
    # Ground truth indices should be the diagonal
    diagonal = torch.arange(similarity.shape[0], device=similarity.device)
    
    # Find rank of ground truth (adding 1 as ranks start from 1)
    ranks = torch.nonzero(indices == diagonal.unsqueeze(-1).expand_as(indices), as_tuple=True)[1] + 1
    
    # Calculate MRR
    mrr = torch.mean(1.0 / ranks.float()).item()
    return mrr

class CrossAttentionReranker:
    """
    Cross-attention reranker for improving retrieval performance.
    
    This reranker uses a cross-attention mechanism to perform fine-grained
    matching between images and captions after initial retrieval.
    """
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device=None):
        """
        Initialize the reranker.
        
        Args:
            model_name: HuggingFace model name for the cross-encoder
            device: Device to run the model on
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.initialized = False
        self.model_name = model_name
        
        # Lazy initialization - load when first used
        self.model = None
        
    def _ensure_initialized(self):
        """Lazy initialization of the model."""
        if not self.initialized:
            try:
                logger.info(f"Initializing cross-attention reranker: {self.model_name}")
                from sentence_transformers import CrossEncoder
                
                self.model = CrossEncoder(self.model_name, device=self.device)
                self.initialized = True
            except Exception as e:
                logger.error(f"Error initializing reranker: {e}")
                raise
    
    def rerank(self, query_images, candidate_texts: List[str], 
               similarity_scores: torch.Tensor, top_k: int = 5) -> Tuple[torch.Tensor, List[int]]:
        """
        Rerank the top-K candidates using cross-attention.
        
        Args:
            query_images: Original query images (not used directly but needed for future extensions)
            candidate_texts: List of all candidate texts
            similarity_scores: Original similarity scores (batch_size x num_candidates)
            top_k: Number of top candidates to rerank
            
        Returns:
            Tuple of (reranked_similarity, reranked_indices)
        """
        self._ensure_initialized()
        
        batch_size = similarity_scores.shape[0]
        reranked_similarity = similarity_scores.clone()
        reranked_indices = []
        
        # For each query
        for i in range(batch_size):
            # Get top-K candidate indices
            _, top_indices = similarity_scores[i].topk(k=min(top_k, len(candidate_texts)))
            top_indices = top_indices.cpu().numpy()
            
            # Get corresponding texts
            top_candidates = [candidate_texts[idx] for idx in top_indices]
            
            # Prepare reranking pairs
            query_text = f"Rerank image results for: image #{i+1}"
            pairs = [(query_text, candidate) for candidate in top_candidates]
            
            try:
                # Get cross-encoder scores
                cross_scores = self.model.predict(pairs)
                
                # Create a tensor to hold reranked scores
                reranked = torch.zeros_like(similarity_scores[i])
                
                # Assign new scores to only the top-K candidates
                for j, (idx, score) in enumerate(zip(top_indices, cross_scores)):
                    reranked[idx] = torch.tensor(score)
                
                # Update the scores
                reranked_similarity[i] = reranked
                
                # Get new ranking from cross-attention scores
                _, new_indices = torch.tensor(cross_scores).sort(descending=True)
                reranked_indices.append([top_indices[idx] for idx in new_indices.tolist()])
            except Exception as e:
                logger.error(f"Error during reranking for query {i}: {e}")
                # Fall back to original ranking
                reranked_indices.append(top_indices.tolist())
        
        return reranked_similarity, reranked_indices

def visualize_comparison(samples, base_similarity, large_similarity, reranked_similarity, 
                        output_dir="results", num_examples=3):
    """
    Visualize examples comparing the base model, large model, and reranker.
    
    Args:
        samples: Dataset samples
        base_similarity: Similarity matrix from base model
        large_similarity: Similarity matrix from large model
        reranked_similarity: Similarity matrix after reranking
        output_dir: Output directory for visualizations
        num_examples: Number of examples to visualize
    """
    logger.info(f"Visualizing {num_examples} comparative examples")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define common color mappings for consistency
    colors = {
        "correct": "lightgreen",
        "incorrect": "white"
    }
    
    for i in range(min(num_examples, len(samples))):
        # Get sample and prepare data
        sample = samples[i]
        image = sample["image"]
        actual_caption = sample["captions"][0]
        
        # Get top 3 matches from each model
        base_scores, base_indices = base_similarity[i].topk(k=3)
        large_scores, large_indices = large_similarity[i].topk(k=3)
        rerank_scores, rerank_indices = reranked_similarity[i].topk(k=3)
        
        # Create a 3-part figure for comparison
        plt.figure(figsize=(18, 8))
        
        # Show image
        plt.subplot(1, 4, 1)
        plt.imshow(image)
        plt.title("Query Image", fontsize=14)
        plt.axis('off')
        
        # Show base model results
        plt.subplot(1, 4, 2)
        plt.axis('off')
        plt.title("Base Model (ViT-B-16-SigLIP)", fontsize=14)
        
        # Add ground truth
        plt.figtext(0.25, 0.85, "Ground Truth:", fontweight='bold', fontsize=11)
        plt.figtext(0.25, 0.82, actual_caption[:50] + "..." if len(actual_caption) > 50 else actual_caption, 
                   fontsize=9, bbox=dict(facecolor='lightblue', alpha=0.3, boxstyle='round,pad=0.5'))
        
        # Add base model matches
        plt.figtext(0.25, 0.78, "Top Matches:", fontweight='bold', fontsize=11)
        
        for j, (idx, score) in enumerate(zip(base_indices.tolist(), base_scores.tolist())):
            matched_caption = samples[idx]["captions"][0]
            short_caption = matched_caption[:50] + "..." if len(matched_caption) > 50 else matched_caption
            color = colors["correct"] if idx == i else colors["incorrect"]
            plt.figtext(0.25, 0.74 - j*0.07, f"{j+1}. Score: {score:.3f}", fontsize=9, fontweight='bold')
            plt.figtext(0.25, 0.71 - j*0.07, short_caption, fontsize=8,
                       bbox=dict(facecolor=color, alpha=0.3, boxstyle='round,pad=0.5'))
        
        # Show large model results
        plt.subplot(1, 4, 3)
        plt.axis('off')
        plt.title("Large Model (ViT-L-16-SigLIP-384)", fontsize=14)
        
        # Add ground truth
        plt.figtext(0.5, 0.85, "Ground Truth:", fontweight='bold', fontsize=11)
        plt.figtext(0.5, 0.82, actual_caption[:50] + "..." if len(actual_caption) > 50 else actual_caption, 
                   fontsize=9, bbox=dict(facecolor='lightblue', alpha=0.3, boxstyle='round,pad=0.5'))
        
        # Add large model matches
        plt.figtext(0.5, 0.78, "Top Matches:", fontweight='bold', fontsize=11)
        
        for j, (idx, score) in enumerate(zip(large_indices.tolist(), large_scores.tolist())):
            matched_caption = samples[idx]["captions"][0]
            short_caption = matched_caption[:50] + "..." if len(matched_caption) > 50 else matched_caption
            color = colors["correct"] if idx == i else colors["incorrect"]
            plt.figtext(0.5, 0.74 - j*0.07, f"{j+1}. Score: {score:.3f}", fontsize=9, fontweight='bold')
            plt.figtext(0.5, 0.71 - j*0.07, short_caption, fontsize=8,
                       bbox=dict(facecolor=color, alpha=0.3, boxstyle='round,pad=0.5'))
        
        # Show reranked results
        plt.subplot(1, 4, 4)
        plt.axis('off')
        plt.title("After Reranking", fontsize=14)
        
        # Add ground truth
        plt.figtext(0.75, 0.85, "Ground Truth:", fontweight='bold', fontsize=11)
        plt.figtext(0.75, 0.82, actual_caption[:50] + "..." if len(actual_caption) > 50 else actual_caption, 
                   fontsize=9, bbox=dict(facecolor='lightblue', alpha=0.3, boxstyle='round,pad=0.5'))
        
        # Add reranked matches
        plt.figtext(0.75, 0.78, "Top Matches:", fontweight='bold', fontsize=11)
        
        for j, (idx, score) in enumerate(zip(rerank_indices.tolist(), rerank_scores.tolist())):
            matched_caption = samples[idx]["captions"][0]
            short_caption = matched_caption[:50] + "..." if len(matched_caption) > 50 else matched_caption
            color = colors["correct"] if idx == i else colors["incorrect"]
            plt.figtext(0.75, 0.74 - j*0.07, f"{j+1}. Score: {score:.3f}", fontsize=9, fontweight='bold')
            plt.figtext(0.75, 0.71 - j*0.07, short_caption, fontsize=8,
                       bbox=dict(facecolor=color, alpha=0.3, boxstyle='round,pad=0.5'))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/comparison_example_{i+1}.png", dpi=150, bbox_inches='tight')
        plt.close()

def plot_comparative_metrics(base_metrics, large_metrics, reranked_metrics, output_dir="results"):
    """
    Create visualization comparing metrics across models.
    
    Args:
        base_metrics: Metrics from base model
        large_metrics: Metrics from large model
        reranked_metrics: Metrics after reranking
        output_dir: Output directory for visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create consolidated dataframe for plotting
    data = []
    
    # Process each metric type
    for metric in ['image_to_text_recall@1', 'image_to_text_recall@5', 'image_to_text_mrr']:
        display_name = metric.replace('image_to_text_', '').upper()
        data.append({"Metric": display_name, "Value": base_metrics[metric], "Model": "ViT-B-16 (Base)"})
        data.append({"Metric": display_name, "Value": large_metrics[metric], "Model": "ViT-L-16-SigLIP-384"})
        data.append({"Metric": display_name, "Value": reranked_metrics[metric], "Model": "With Reranking"})
    
    df = pd.DataFrame(data)
    
    # Create comparative bar chart
    plt.figure(figsize=(12, 8))
    chart = sns.barplot(x="Metric", y="Value", hue="Model", data=df)
    chart.set_ylim(0, 1.1)
    
    # Add value labels on bars
    for container in chart.containers:
        chart.bar_label(container, fmt='%.3f', padding=3)
    
    plt.title("SigLIP Model Size Comparison", fontsize=16)
    plt.xlabel("Metric", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.legend(title="Model Variant")
    plt.tight_layout()
    
    # Save chart
    plt.savefig(f"{output_dir}/model_comparison_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save metrics as CSV
    csv_path = os.path.join(output_dir, "comparison_metrics.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved comparison metrics to {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Comparative evaluation of SigLIP models")
    
    # Dataset settings
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to use (default: 100, 0 for all)")
    parser.add_argument("--cache_dir", type=str, default="./dataset_cache",
                       help="Cache directory for datasets (default: ./dataset_cache)")
    
    # Model settings
    parser.add_argument("--base_model", type=str, default="ViT-B-16-SigLIP",
                       help="Base SigLIP model architecture (default: ViT-B-16-SigLIP)")
    parser.add_argument("--large_model", type=str, default="ViT-L-16-SigLIP-384",
                       help="Large SigLIP model architecture (default: ViT-L-16-SigLIP-384)")
    parser.add_argument("--pretrained", type=str, default="webli",
                       help="Pretrained weights identifier (default: webli)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to run on (default: auto-detect)")
    parser.add_argument("--precision", type=str, default="fp16", 
                       choices=["fp32", "fp16", "bf16", "int8"],
                       help="Inference precision (default: fp16)")
    
    # Reranking settings
    parser.add_argument("--reranker_model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                       help="Cross-encoder model for reranking (default: cross-encoder/ms-marco-MiniLM-L-6-v2)")
    parser.add_argument("--top_k", type=int, default=10,
                       help="Top-K results to rerank (default: 10)")
    
    # Batch settings
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size (default: 32)")
    
    # Output settings
    parser.add_argument("--output_dir", type=str, default="./comparison_results",
                       help="Output directory (default: ./comparison_results)")
    
    args = parser.parse_args()
    
    # Convert num_samples=0 to None (use all samples)
    num_samples = None if args.num_samples == 0 else args.num_samples
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log configuration
    logger.info("Starting SigLIP model comparison with configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Load dataset samples
    samples = load_flickr30k_dataset(
        num_samples=num_samples,
        cache_dir=args.cache_dir
    )
    
    # Initialize base SigLIP model
    logger.info(f"Initializing base SigLIP model: {args.base_model}...")
    base_model = create_advanced_siglip_model(
        model_name=args.base_model,
        pretrained=args.pretrained,
        device=args.device,
        precision=args.precision,
        initial_batch_size=max(1, args.batch_size // 2),
        max_batch_size=args.batch_size
    )
    
    # Process images with base model
    logger.info("Encoding images with base model...")
    base_img_embeddings_result = base_model.encode_images(
        [sample["image"] for sample in samples],
        batch_size=args.batch_size
    )
    base_img_embeddings = base_img_embeddings_result["image_embeddings"]
    
    # Process captions with base model
    logger.info("Encoding text with base model...")
    base_txt_embeddings_result = base_model.encode_text(
        [sample["captions"][0] for sample in samples],
        batch_size=args.batch_size
    )
    base_txt_embeddings = base_txt_embeddings_result["text_embeddings"]
    
    # Compute similarity with base model
    logger.info("Computing similarity matrix for base model...")
    base_similarity = base_model.compute_similarity(base_img_embeddings, base_txt_embeddings)
    
    # Compute metrics for base model
    logger.info("Computing retrieval metrics for base model...")
    base_metrics = compute_retrieval_metrics(base_similarity)
    logger.info("Base model metrics:")
    for name, value in base_metrics.items():
        logger.info(f"  {name}: {value:.4f}")
    
    # Initialize large SigLIP model - use much smaller batch sizes for large model
    logger.info(f"Initializing large SigLIP model: {args.large_model}...")
    large_model = create_advanced_siglip_model(
        model_name=args.large_model,
        pretrained=args.pretrained,
        device=args.device,
        precision=args.precision,
        initial_batch_size=2,  # Start with a very small batch size
        max_batch_size=8       # Cap at a reasonable size for large model
    )
    
    # Process images with large model
    logger.info("Encoding images with large model...")
    large_img_embeddings_result = large_model.encode_images(
        [sample["image"] for sample in samples],
        batch_size=4  # Use small fixed batch size
    )
    large_img_embeddings = large_img_embeddings_result["image_embeddings"]
    
    # Process captions with large model
    logger.info("Encoding text with large model...")
    large_txt_embeddings_result = large_model.encode_text(
        [sample["captions"][0] for sample in samples],
        batch_size=8  # Text encoding is less memory intensive
    )
    large_txt_embeddings = large_txt_embeddings_result["text_embeddings"]
    
    # Compute similarity with large model
    logger.info("Computing similarity matrix for large model...")
    large_similarity = large_model.compute_similarity(large_img_embeddings, large_txt_embeddings)
    
    # Compute metrics for large model
    logger.info("Computing retrieval metrics for large model...")
    large_metrics = compute_retrieval_metrics(large_similarity)
    logger.info("Large model metrics:")
    for name, value in large_metrics.items():
        logger.info(f"  {name}: {value:.4f}")
    
    # Initialize reranker
    logger.info(f"Initializing cross-attention reranker...")
    reranker = CrossAttentionReranker(model_name=args.reranker_model, device=args.device)
    
    # Apply reranking to large model results
    logger.info(f"Applying reranking to top-{args.top_k} results...")
    captions = [sample["captions"][0] for sample in samples]
    reranked_similarity, _ = reranker.rerank(
        query_images=[sample["image"] for sample in samples],
        candidate_texts=captions,
        similarity_scores=large_similarity,
        top_k=args.top_k
    )
    
    # Compute metrics for reranked results
    logger.info("Computing retrieval metrics after reranking...")
    reranked_metrics = compute_retrieval_metrics(reranked_similarity)
    logger.info("Reranked metrics:")
    for name, value in reranked_metrics.items():
        logger.info(f"  {name}: {value:.4f}")
    
    # Save embeddings and metadata for LLaVA reasoning
    embeddings_dir = os.path.join(args.output_dir, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Save image embeddings from both models
    logger.info("Saving embeddings for LLaVA reasoning...")
    
    # Save base model embeddings using NumPy format (.npy)
    base_img_embeddings_path = os.path.join(embeddings_dir, "base_img_embeddings.npy")
    base_txt_embeddings_path = os.path.join(embeddings_dir, "base_txt_embeddings.npy")
    np.save(base_img_embeddings_path, base_img_embeddings.cpu().numpy())
    np.save(base_txt_embeddings_path, base_txt_embeddings.cpu().numpy())
    logger.info(f"Saved base model embeddings to {base_img_embeddings_path} and {base_txt_embeddings_path}")
    
    # Save large model embeddings using NumPy format (.npy)
    large_img_embeddings_path = os.path.join(embeddings_dir, "large_img_embeddings.npy")
    large_txt_embeddings_path = os.path.join(embeddings_dir, "large_txt_embeddings.npy")
    np.save(large_img_embeddings_path, large_img_embeddings.cpu().numpy())
    np.save(large_txt_embeddings_path, large_txt_embeddings.cpu().numpy())
    logger.info(f"Saved large model embeddings to {large_img_embeddings_path} and {large_txt_embeddings_path}")
    
    # Save metadata separately using pickle (.pkl) for complex Python objects
    import pickle
    metadata_path = os.path.join(embeddings_dir, "metadata.pkl")
    metadata = {
        "captions": captions,
        "image_paths": [getattr(sample["image"], "filename", f"image_{i}") for i, sample in enumerate(samples)],
        "similarity": {
            "base": base_similarity.cpu().numpy(),
            "large": large_similarity.cpu().numpy(),
            "reranked": reranked_similarity.cpu().numpy()
        },
        "metrics": {
            "base": base_metrics,
            "large": large_metrics,
            "reranked": reranked_metrics
        },
        "models": {
            "base": {
                "name": args.base_model,
                "pretrained": args.pretrained,
                "precision": args.precision,
                "embedding_dim": base_img_embeddings.shape[1]
            },
            "large": {
                "name": args.large_model,
                "pretrained": args.pretrained,
                "precision": args.precision,
                "embedding_dim": large_img_embeddings.shape[1]
            }
        }
    }
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    logger.info(f"Saved metadata to {metadata_path}")
    
    # Keep JSON format for easy parsing by external systems
    import json
    json_metadata = {
        "captions": captions,
        "image_paths": [getattr(sample["image"], "filename", f"image_{i}") for i, sample in enumerate(samples)],
        "models": {
            "base": {
                "name": args.base_model,
                "pretrained": args.pretrained,
                "embedding_dim": base_img_embeddings.shape[1]
            },
            "large": {
                "name": args.large_model,
                "pretrained": args.pretrained,
                "embedding_dim": large_img_embeddings.shape[1]
            }
        }
    }
    
    with open(os.path.join(embeddings_dir, "metadata.json"), "w") as f:
        json.dump(json_metadata, f, indent=2)
    
    logger.info(f"Saved all embeddings and metadata to {embeddings_dir}")
    
    # Visualize comparison examples
    visualize_comparison(
        samples=samples,
        base_similarity=base_similarity,
        large_similarity=large_similarity,
        reranked_similarity=reranked_similarity,
        output_dir=args.output_dir,
        num_examples=5
    )
    
    # Plot comparative metrics
    plot_comparative_metrics(
        base_metrics=base_metrics,
        large_metrics=large_metrics,
        reranked_metrics=reranked_metrics,
        output_dir=args.output_dir
    )
    
    logger.info(f"Model comparison completed! Results saved to {args.output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 



    """
    Image Embeddings in NumPy Format:
base_img_embeddings.npy (30KB) - Base model (ViT-B-16) image embeddings
large_img_embeddings.npy (40KB) - Large model (ViT-L-16) image embeddings with higher dimensionality
Text Embeddings in NumPy Format:
base_txt_embeddings.npy (30KB) - Base model text embeddings
large_txt_embeddings.npy (40KB) - Large model text embeddings
Metadata in Multiple Formats:
metadata.pkl (3.3KB) - Complete metadata in Python pickle format for ML libraries
metadata.json (1.3KB) - Simplified metadata in JSON for easy parsing
The embedding files are now organized in a clean, flat structure with industry-standard formats that are optimized for:
Fast loading with NumPy
Complete data preservation with pickle
Easy parsing with JSON
Maximum compatibility with other ML and reasoning systems
    """