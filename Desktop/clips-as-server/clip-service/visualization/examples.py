#!/usr/bin/env python3
"""
Example usage of the EmbeddingVisualizer class.

This script demonstrates various ways to use the EmbeddingVisualizer
to generate, visualize, and search CLIP embeddings.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from embedding_visualizer import EmbeddingVisualizer
from input.dataset_config import DatasetConfig


async def example_1_generate_and_visualize():
    """Generate embeddings and visualize them with t-SNE."""
    print("Example 1: Generate embeddings and visualize with t-SNE")
    
    # Initialize visualizer
    visualizer = EmbeddingVisualizer(
        model_name="ViT-B/32",
        output_dir="output/examples/example1"
    )
    
    # Create dataset config
    dataset_config = DatasetConfig(
        image_datasets=["cifar100"],  # Using CIFAR-100 for this example
        text_datasets=["cifar100"],
        batch_size=32
    )
    
    # Generate and save embeddings
    await visualizer.generate_and_save_embeddings(
        dataset_config,
        max_samples=200,  # Limit to 200 samples for this example
        save_filename="cifar100_embeddings.pkl"
    )
    
    # Generate t-SNE visualization
    visualizer.visualize_tsne(
        output_filename="cifar100_tsne.png"
    )
    
    print("Example 1 completed. Check output/examples/example1 for results.")


async def example_2_text_image_search():
    """Load embeddings and perform text-to-image search."""
    print("Example 2: Load embeddings and perform text-to-image search")
    
    # Initialize visualizer
    visualizer = EmbeddingVisualizer(
        model_name="ViT-B/32",
        output_dir="output/examples/example2"
    )
    
    # Path to embeddings from Example 1
    embeddings_path = Path("output/examples/example1/cifar100_embeddings.pkl")
    
    # Check if embeddings exist, otherwise generate them
    if not embeddings_path.exists():
        print("Embeddings not found. Generating them first...")
        # Create dataset config
        dataset_config = DatasetConfig(
            image_datasets=["cifar100"],
            text_datasets=["cifar100"],
            batch_size=32
        )
        
        # Ensure directory exists
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate and save embeddings
        await visualizer.generate_and_save_embeddings(
            dataset_config,
            max_samples=200,
            save_filename=str(embeddings_path)
        )
    else:
        # Load existing embeddings
        visualizer.load_embeddings(str(embeddings_path))
    
    # Perform text-to-image search with various queries
    queries = [
        "an airplane in the sky",
        "a dog",
        "a red car",
        "a beautiful flower"
    ]
    
    for i, query in enumerate(queries):
        visualizer.visualize_text_to_image_search(
            query_text=query,
            k=5,  # Show top 5 results
            output_filename=f"search_result_{i}.png"
        )
    
    print("Example 2 completed. Check output/examples/example2 for results.")


async def example_3_umap_visualization():
    """Load embeddings and create UMAP visualization."""
    print("Example 3: Load embeddings and create UMAP visualization")
    
    # Initialize visualizer
    visualizer = EmbeddingVisualizer(
        model_name="ViT-B/32",
        output_dir="output/examples/example3"
    )
    
    # Path to embeddings from Example 1
    embeddings_path = Path("output/examples/example1/cifar100_embeddings.pkl")
    
    # Check if embeddings exist, otherwise generate them
    if not embeddings_path.exists():
        print("Embeddings not found. Generating them first...")
        # Create dataset config
        dataset_config = DatasetConfig(
            image_datasets=["cifar100"],
            text_datasets=["cifar100"],
            batch_size=32
        )
        
        # Ensure directory exists
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate and save embeddings
        await visualizer.generate_and_save_embeddings(
            dataset_config,
            max_samples=200,
            save_filename=str(embeddings_path)
        )
    else:
        # Load existing embeddings
        visualizer.load_embeddings(str(embeddings_path))
    
    # Create UMAP visualization with different parameters
    visualizer.visualize_umap(
        n_neighbors=15,
        min_dist=0.1,
        output_filename="umap_default.png"
    )
    
    visualizer.visualize_umap(
        n_neighbors=30,
        min_dist=0.05,
        output_filename="umap_detailed.png"
    )
    
    visualizer.visualize_umap(
        n_neighbors=5,
        min_dist=0.5,
        output_filename="umap_spread.png"
    )
    
    print("Example 3 completed. Check output/examples/example3 for results.")


# Run all examples
async def main():
    print("Running CLIP Embedding Visualizer Examples")
    
    # Ensure output directories exist
    Path("output/examples/example1").mkdir(parents=True, exist_ok=True)
    Path("output/examples/example2").mkdir(parents=True, exist_ok=True)
    Path("output/examples/example3").mkdir(parents=True, exist_ok=True)
    
    # Run examples
    await example_1_generate_and_visualize()
    await example_2_text_image_search()
    await example_3_umap_visualization()
    
    print("All examples completed successfully.")


if __name__ == "__main__":
    asyncio.run(main()) 