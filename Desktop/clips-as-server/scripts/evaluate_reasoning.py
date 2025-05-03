#!/usr/bin/env python3
"""
Evaluation script for analyzing the quality of LLaVA/Qwen reasoning results.
This script computes various metrics to assess the quality of reasoning:
- Cosine similarity between embeddings
- Recall@K for ranking tasks
- Mean Reciprocal Rank (MRR) for ranking accuracy
- Embedding quality analysis to detect exploding/vanishing gradients

Usage:
python scripts/evaluate_reasoning.py --enriched-data llava_reasoning.json --output-file reasoning_metrics.json
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_enriched_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load enriched data from JSON or pickle file.
    
    Args:
        file_path: Path to JSON or pickle file with LLaVA/Qwen enriched data
        
    Returns:
        List of enriched metadata dictionaries
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
    elif file_path.suffix == '.pkl':
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    logger.info(f"Loaded {len(data)} entries from {file_path}")
    return data

def extract_embeddings(data: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """
    Extract embeddings from enriched data.
    
    Args:
        data: Enriched metadata with embeddings
        
    Returns:
        Dictionary with extracted embeddings
    """
    # Determine embedding types in the data
    embedding_types = {}
    first_entry = data[0]
    
    for key in first_entry:
        if key.endswith('_img_embedding') or key.endswith('_txt_embedding'):
            embedding_types[key] = []
    
    # Extract embeddings
    for entry in data:
        for embed_type in embedding_types:
            if embed_type in entry:
                # Convert to numpy array if it's a list
                if isinstance(entry[embed_type], list):
                    embedding_types[embed_type].append(np.array(entry[embed_type]))
                else:
                    embedding_types[embed_type].append(entry[embed_type])
    
    # Convert lists to numpy arrays
    for embed_type in embedding_types:
        embedding_types[embed_type] = np.array(embedding_types[embed_type])
        
    logger.info(f"Extracted {len(embedding_types)} embedding types")
    for embed_type, embeds in embedding_types.items():
        logger.info(f"  {embed_type}: {embeds.shape}")
    
    return embedding_types

def calculate_cosine_similarities(embeddings: Dict[str, np.ndarray], 
                                reference_idx: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Calculate cosine similarities between embeddings.
    
    Args:
        embeddings: Dictionary with embeddings
        reference_idx: Optional index to use as reference point (defaults to first embedding)
        
    Returns:
        Dictionary with cosine similarities
    """
    if reference_idx is None:
        reference_idx = 0
    
    logger.info(f"Calculating cosine similarities using reference index {reference_idx}")
    similarities = {}
    
    for embed_type, embeds in embeddings.items():
        if len(embeds) <= reference_idx:
            logger.warning(f"Reference index {reference_idx} is out of bounds for {embed_type}")
            continue
            
        reference = embeds[reference_idx].reshape(1, -1)
        similarities[embed_type] = cosine_similarity(embeds, reference).flatten()
        
    return similarities

def compute_recall_at_k(similarities: Dict[str, np.ndarray], 
                       ground_truth_idx: List[int],
                       k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Dict[int, float]]:
    """
    Compute Recall@K for each embedding type.
    
    Args:
        similarities: Dictionary with cosine similarities
        ground_truth_idx: Indices of ground truth entries
        k_values: List of K values to compute Recall@K for
        
    Returns:
        Dictionary with Recall@K values for each embedding type and K
    """
    logger.info(f"Computing Recall@K for k values: {k_values}")
    
    recall_at_k = {}
    
    for embed_type, sims in similarities.items():
        recall_at_k[embed_type] = {}
        
        # Get indices sorted by similarity (descending)
        sorted_indices = np.argsort(-sims)
        
        for k in k_values:
            if k > len(sorted_indices):
                logger.warning(f"K value {k} is larger than number of items {len(sorted_indices)}")
                recall_at_k[embed_type][k] = None
                continue
                
            # Check if any ground truth is in top-K
            top_k_indices = sorted_indices[:k]
            hits = sum(1 for idx in ground_truth_idx if idx in top_k_indices)
            recall = hits / len(ground_truth_idx) if ground_truth_idx else 0
            
            recall_at_k[embed_type][k] = recall
    
    return recall_at_k

def compute_mrr(similarities: Dict[str, np.ndarray], 
               ground_truth_idx: List[int]) -> Dict[str, float]:
    """
    Compute Mean Reciprocal Rank (MRR) for each embedding type.
    
    Args:
        similarities: Dictionary with cosine similarities
        ground_truth_idx: Indices of ground truth entries
        
    Returns:
        Dictionary with MRR values for each embedding type
    """
    logger.info("Computing Mean Reciprocal Rank (MRR)")
    
    mrr_values = {}
    
    for embed_type, sims in similarities.items():
        # Get indices sorted by similarity (descending)
        sorted_indices = np.argsort(-sims)
        
        # Calculate reciprocal rank for each ground truth
        reciprocal_ranks = []
        for gt_idx in ground_truth_idx:
            # Find the rank of the ground truth
            rank = np.where(sorted_indices == gt_idx)[0][0] + 1 if gt_idx in sorted_indices else 0
            
            # Calculate reciprocal rank
            rr = 1.0 / rank if rank > 0 else 0
            reciprocal_ranks.append(rr)
        
        # Calculate MRR
        mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
        mrr_values[embed_type] = mrr
    
    return mrr_values

def analyze_embedding_quality(embeddings: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Analyze embedding quality to detect exploding/vanishing gradients.
    
    Args:
        embeddings: Dictionary with embeddings
        
    Returns:
        Dictionary with statistics for each embedding type
    """
    logger.info("Analyzing embedding quality")
    
    quality_metrics = {}
    
    for embed_type, embeds in embeddings.items():
        # Calculate metrics
        metrics = {
            'mean': float(np.mean(embeds)),
            'std': float(np.std(embeds)),
            'min': float(np.min(embeds)),
            'max': float(np.max(embeds)),
            'norm_mean': float(np.mean(np.linalg.norm(embeds, axis=1))),
            'zero_fraction': float(np.mean(embeds == 0)),
        }
        
        # Check for exploding/vanishing values
        metrics['has_exploding'] = abs(metrics['max']) > 100 or abs(metrics['min']) > 100
        metrics['has_vanishing'] = metrics['std'] < 0.01 or metrics['zero_fraction'] > 0.5
        
        quality_metrics[embed_type] = metrics
    
    return quality_metrics

def analyze_reasoning_outputs(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze LLaVA/Qwen reasoning outputs.
    
    Args:
        data: Enriched metadata with reasoning outputs
        
    Returns:
        Dictionary with analysis results
    """
    logger.info("Analyzing reasoning outputs")
    
    # Identify reasoning fields
    reasoning_fields = []
    first_entry = data[0]
    
    for key in first_entry:
        if (key.startswith('llava_') or key.startswith('qwen_')) and not key.endswith('_embedding'):
            reasoning_fields.append(key)
    
    logger.info(f"Found reasoning fields: {reasoning_fields}")
    
    # Analyze reasoning output length and content
    reasoning_metrics = {}
    
    for field in reasoning_fields:
        # Extract reasoning outputs
        outputs = [entry[field] for entry in data if field in entry]
        
        # Calculate metrics
        metrics = {
            'count': len(outputs),
            'avg_length': np.mean([len(output) for output in outputs]) if outputs else 0,
            'min_length': min([len(output) for output in outputs]) if outputs else 0,
            'max_length': max([len(output) for output in outputs]) if outputs else 0,
            'error_count': sum(1 for output in outputs if output.startswith('Error:') or output == 'No image available for processing')
        }
        
        reasoning_metrics[field] = metrics
    
    return reasoning_metrics

def plot_similarity_distribution(similarities: Dict[str, np.ndarray], 
                                output_dir: str = '.'):
    """
    Plot distribution of cosine similarities.
    
    Args:
        similarities: Dictionary with cosine similarities
        output_dir: Directory to save plots
    """
    logger.info(f"Plotting similarity distributions to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    for embed_type, sims in similarities.items():
        plt.figure(figsize=(10, 6))
        plt.hist(sims, bins=20, alpha=0.7)
        plt.title(f'Cosine Similarity Distribution - {embed_type}')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'similarity_{embed_type}.png'))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLaVA/Qwen reasoning results")
    parser.add_argument("--enriched-data", type=str, required=True,
                      help="Path to JSON or pickle file with enriched data")
    parser.add_argument("--output-file", type=str, default="reasoning_metrics.json",
                      help="Output file for evaluation metrics")
    parser.add_argument("--reference-idx", type=int, default=0,
                      help="Index to use as reference for similarity calculations")
    parser.add_argument("--ground-truth-idx", type=str, default="0",
                      help="Comma-separated indices of ground truth entries")
    parser.add_argument("--plot-similarities", action="store_true",
                      help="Plot similarity distributions")
    parser.add_argument("--plot-dir", type=str, default="similarity_plots",
                      help="Directory to save similarity plots")
    parser.add_argument("--k-values", type=str, default="1,3,5,10",
                      help="Comma-separated values of K for Recall@K")
    
    args = parser.parse_args()
    
    # Parse K values and ground truth indices
    k_values = [int(k) for k in args.k_values.split(',')]
    ground_truth_idx = [int(idx) for idx in args.ground_truth_idx.split(',')]
    
    # Load data
    data = load_enriched_data(args.enriched_data)
    
    # Extract embeddings
    embeddings = extract_embeddings(data)
    
    # Calculate cosine similarities
    similarities = calculate_cosine_similarities(embeddings, args.reference_idx)
    
    # Compute Recall@K
    recall_at_k = compute_recall_at_k(similarities, ground_truth_idx, k_values)
    
    # Compute MRR
    mrr = compute_mrr(similarities, ground_truth_idx)
    
    # Analyze embedding quality
    quality_metrics = analyze_embedding_quality(embeddings)
    
    # Analyze reasoning outputs
    reasoning_metrics = analyze_reasoning_outputs(data)
    
    # Create output metrics
    metrics = {
        'data_file': args.enriched_data,
        'num_entries': len(data),
        'reference_idx': args.reference_idx,
        'ground_truth_idx': ground_truth_idx,
        'recall_at_k': recall_at_k,
        'mrr': mrr,
        'embedding_quality': quality_metrics,
        'reasoning_metrics': reasoning_metrics
    }
    
    # Save metrics
    with open(args.output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Saved evaluation metrics to {args.output_file}")
    
    # Plot similarity distributions if requested
    if args.plot_similarities:
        plot_similarity_distribution(similarities, args.plot_dir)
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Data file: {args.enriched_data}")
    print(f"Number of entries: {len(data)}")
    
    print("\nRecall@K:")
    for embed_type, recalls in recall_at_k.items():
        print(f"  {embed_type}:")
        for k, recall in recalls.items():
            print(f"    R@{k}: {recall:.4f}")
    
    print("\nMean Reciprocal Rank (MRR):")
    for embed_type, value in mrr.items():
        print(f"  {embed_type}: {value:.4f}")
    
    print("\nEmbedding Quality:")
    for embed_type, metrics in quality_metrics.items():
        print(f"  {embed_type}:")
        print(f"    Mean: {metrics['mean']:.4f}, Std: {metrics['std']:.4f}")
        print(f"    Min: {metrics['min']:.4f}, Max: {metrics['max']:.4f}")
        if metrics['has_exploding']:
            print("    WARNING: Potentially exploding embeddings")
        if metrics['has_vanishing']:
            print("    WARNING: Potentially vanishing embeddings")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 