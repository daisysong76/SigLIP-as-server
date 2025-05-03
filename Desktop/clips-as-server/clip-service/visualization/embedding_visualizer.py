#!/usr/bin/env python3
"""
Embedding Visualization and Analysis Tool for CLIP

This script provides functionality to:
1. Save embeddings and their corresponding IDs
2. Visualize embeddings using t-SNE and UMAP
3. Perform and visualize top-K nearest neighbor searches
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import clip
from PIL import Image
import requests
from io import BytesIO
from pathlib import Path
import argparse
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.manifold import TSNE
import umap
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.collate_fn import multimodal_collate_fn
from transformers import CLIPProcessor
from datasets import load_dataset
from input.dataset_config import DatasetConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmbeddingVisualizer:
    """Tool for visualizing and analyzing CLIP embeddings."""
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        output_dir: str = "output/visualizations",
        device: Optional[torch.device] = None
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and processor
        logger.info(f"Loading CLIP model: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            torch_dtype=torch.float32
        )
        
        # Initialize storage for embeddings
        self.image_embeddings = None
        self.text_embeddings = None
        self.image_ids = None
        self.text_ids = None
        self.image_paths = None
        self.texts = None
    
    async def generate_and_save_embeddings(
        self,
        dataset_config: DatasetConfig,
        max_samples: int = 500,
        save_filename: str = "embeddings.pkl"
    ) -> Dict[str, Any]:
        """Generate and save embeddings from a dataset."""
        logger.info(f"Generating embeddings for up to {max_samples} samples")
        
        # Load dataset
        image_dataset, text_dataset = await dataset_config.load_streaming_dataset(
            self.processor,
            streaming=True
        )
        
        # Collect embeddings
        image_embeddings = []
        text_embeddings = []
        image_ids = []
        text_ids = []
        image_data = []  # Store actual images or paths
        text_data = []   # Store actual text strings
        
        # Process images
        image_count = 0
        async for batch in image_dataset:
            # Process batch
            pixel_values = batch["pixel_values"]
            
            # Ensure pixel_values is a tensor, not a list
            if isinstance(pixel_values, list):
                pixel_values = torch.stack(pixel_values)
            
            pixel_values = pixel_values.to(torch.float32)
            
            # Add batch dimension if needed
            if pixel_values.ndim == 3:
                pixel_values = pixel_values.unsqueeze(0)
            
            # Generate embeddings
            with torch.no_grad():
                pixel_values = pixel_values.to(self.device)
                features = self.model.encode_image(pixel_values)
                features = features.to(torch.float32)
                features = torch.nn.functional.normalize(features, p=2, dim=-1)
                image_embeddings.append(features.detach().cpu().numpy())
            
            # Store IDs
            id_key = "image_ids" if "image_ids" in batch else "image_id"
            if id_key in batch:
                ids = batch[id_key] if isinstance(batch[id_key], list) else [batch[id_key]]
                image_ids.extend(ids)
            
            # Store original image data if available (for visualization)
            if "image" in batch:
                image_data.extend(batch["image"] if isinstance(batch["image"], list) else [batch["image"]])
            
            # Update count
            image_count += len(pixel_values)
            if image_count >= max_samples:
                break
        
        # Process text
        text_count = 0
        async for batch in text_dataset:
            try:
                # Ensure we have text data to tokenize
                if "text" in batch:
                    texts = batch["text"] if isinstance(batch["text"], list) else [batch["text"]]
                    
                    # Properly tokenize the text
                    inputs = self.processor(
                        text=texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    )
                    input_ids = inputs.input_ids.to(self.device)
                    
                    # Generate embeddings
                    with torch.no_grad():
                        features = self.model.encode_text(input_ids)
                        features = features.to(torch.float32)
                        features = torch.nn.functional.normalize(features, p=2, dim=-1)
                        text_embeddings.append(features.detach().cpu().numpy())
                    
                    # Store text data for later use
                    text_data.extend(texts)
                    
                    # Store IDs
                    id_key = "text_ids" if "text_ids" in batch else "text_id"
                    if id_key in batch:
                        ids = batch[id_key] if isinstance(batch[id_key], list) else [batch[id_key]]
                        text_ids.extend(ids)
                    else:
                        # Generate sequential IDs if none provided
                        text_ids.extend(list(range(text_count, text_count + len(texts))))
                    
                    # Update count
                    text_count += len(texts)
                    if text_count >= max_samples:
                        break
                else:
                    # If batch already has input_ids and no text
                    input_ids = batch["input_ids"].to(self.device)
                    
                    # Generate embeddings
                    with torch.no_grad():
                        features = self.model.encode_text(input_ids)
                        features = features.to(torch.float32)
                        features = torch.nn.functional.normalize(features, p=2, dim=-1)
                        text_embeddings.append(features.detach().cpu().numpy())
                    
                    # Store IDs
                    id_key = "text_ids" if "text_ids" in batch else "text_id"
                    if id_key in batch:
                        ids = batch[id_key] if isinstance(batch[id_key], list) else [batch[id_key]]
                        text_ids.extend(ids)
                    
                    # Update count
                    text_count += len(input_ids)
                    if text_count >= max_samples:
                        break
            except Exception as e:
                logger.error(f"Error processing text batch: {e}")
                continue
        
        # Combine embeddings and save
        logger.info(f"Collected {len(image_embeddings)} image batches and {len(text_embeddings)} text batches")
        
        # Concatenate results
        self.image_embeddings = np.concatenate(image_embeddings, axis=0) if image_embeddings else np.array([])
        self.text_embeddings = np.concatenate(text_embeddings, axis=0) if text_embeddings else np.array([])
        self.image_ids = image_ids
        self.text_ids = text_ids
        self.image_paths = image_data
        self.texts = text_data
        
        logger.info(f"Final embeddings shapes: Images {self.image_embeddings.shape}, Texts {self.text_embeddings.shape}")
        
        # Save to file
        save_path = self.output_dir / save_filename
        with open(save_path, 'wb') as f:
            pickle.dump({
                'image_embeddings': self.image_embeddings,
                'text_embeddings': self.text_embeddings,
                'image_ids': self.image_ids,
                'text_ids': self.text_ids,
                'image_paths': self.image_paths,
                'texts': self.texts,
                'model_name': self.model_name
            }, f)
        
        logger.info(f"Saved embeddings to {save_path}")
        return {
            'image_embeddings': self.image_embeddings,
            'text_embeddings': self.text_embeddings,
            'image_count': len(self.image_ids),
            'text_count': len(self.text_ids)
        }
    
    def load_embeddings(self, filename: str) -> None:
        """Load embeddings from a file."""
        load_path = self.output_dir / filename
        logger.info(f"Loading embeddings from {load_path}")
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.image_embeddings = data['image_embeddings']
        self.text_embeddings = data['text_embeddings']
        self.image_ids = data['image_ids']
        self.text_ids = data['text_ids']
        self.image_paths = data.get('image_paths', [])
        self.texts = data.get('texts', [])
        
        logger.info(f"Loaded embeddings: Images {self.image_embeddings.shape}, Texts {self.text_embeddings.shape}")
    
    def visualize_tsne(
        self,
        n_components: int = 2,
        perplexity: int = 30,
        learning_rate: Union[str, float] = 'auto',
        output_filename: str = "tsne_visualization.png"
    ) -> None:
        """Visualize embeddings using t-SNE."""
        if self.image_embeddings is None or self.text_embeddings is None:
            logger.error("No embeddings loaded. Load or generate embeddings first.")
            return
        
        logger.info("Preparing t-SNE visualization")
        
        # Combine embeddings for joint visualization
        combined_embeddings = np.vstack([self.image_embeddings, self.text_embeddings])
        # Labels for coloring points (0 for images, 1 for text)
        labels = np.array([0] * len(self.image_embeddings) + [1] * len(self.text_embeddings))
        
        # Apply t-SNE
        logger.info(f"Running t-SNE with perplexity={perplexity}")
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            init='pca',
            random_state=42
        )
        embeddings_2d = tsne.fit_transform(combined_embeddings)
        
        # Split back into image and text embeddings
        image_embeddings_2d = embeddings_2d[:len(self.image_embeddings)]
        text_embeddings_2d = embeddings_2d[len(self.image_embeddings):]
        
        # Visualize
        plt.figure(figsize=(12, 10))
        plt.scatter(
            image_embeddings_2d[:, 0], image_embeddings_2d[:, 1],
            c='blue', label='Images', alpha=0.5, s=5
        )
        plt.scatter(
            text_embeddings_2d[:, 0], text_embeddings_2d[:, 1],
            c='red', label='Text', alpha=0.5, s=5
        )
        
        plt.title(f't-SNE Visualization of CLIP Embeddings ({self.model_name})')
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / output_filename
        plt.savefig(save_path, dpi=300)
        logger.info(f"Saved t-SNE visualization to {save_path}")
        
        # Also display the plot if run in an interactive environment
        plt.show()

    def visualize_umap(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        output_filename: str = "umap_visualization.png"
    ) -> None:
        """Visualize embeddings using UMAP."""
        if self.image_embeddings is None or self.text_embeddings is None:
            logger.error("No embeddings loaded. Load or generate embeddings first.")
            return
        
        logger.info("Preparing UMAP visualization")
        
        # Combine embeddings for joint visualization
        combined_embeddings = np.vstack([self.image_embeddings, self.text_embeddings])
        
        # Apply UMAP
        logger.info(f"Running UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}")
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42
        )
        embeddings_2d = reducer.fit_transform(combined_embeddings)
        
        # Split back into image and text embeddings
        image_embeddings_2d = embeddings_2d[:len(self.image_embeddings)]
        text_embeddings_2d = embeddings_2d[len(self.image_embeddings):]
        
        # Visualize
        plt.figure(figsize=(12, 10))
        plt.scatter(
            image_embeddings_2d[:, 0], image_embeddings_2d[:, 1],
            c='blue', label='Images', alpha=0.5, s=5
        )
        plt.scatter(
            text_embeddings_2d[:, 0], text_embeddings_2d[:, 1],
            c='red', label='Text', alpha=0.5, s=5
        )
        
        plt.title(f'UMAP Visualization of CLIP Embeddings ({self.model_name})')
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / output_filename
        plt.savefig(save_path, dpi=300)
        logger.info(f"Saved UMAP visualization to {save_path}")
        
        # Also display the plot if run in an interactive environment
        plt.show()
    
    def find_nearest_neighbors(
        self,
        query_embedding: np.ndarray,
        reference_embeddings: np.ndarray,
        reference_ids: List,
        k: int = 5
    ) -> Tuple[List[int], np.ndarray]:
        """Find k nearest neighbors to the query embedding."""
        # Normalize query embedding if needed
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Compute cosine similarities
        similarities = np.dot(reference_embeddings, query_embedding)
        
        # Find top k indices
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Get IDs and similarities
        top_ids = [reference_ids[i] for i in top_indices]
        top_similarities = similarities[top_indices]
        
        return top_ids, top_similarities
    
    def visualize_text_to_image_search(
        self,
        query_text: str,
        k: int = 5,
        output_filename: Optional[str] = None
    ) -> None:
        """Visualize top-k images for a text query."""
        if self.image_embeddings is None:
            logger.error("No image embeddings loaded. Load or generate embeddings first.")
            return
        
        logger.info(f"Performing text-to-image search for: '{query_text}'")
        
        # Generate text embedding
        with torch.no_grad():
            text_inputs = self.processor(
                text=[query_text],
                return_tensors="pt",
                padding=True
            )
            text_embedding = self.model.encode_text(text_inputs.input_ids.to(self.device))
            text_embedding = text_embedding.to(torch.float32)
            text_embedding = torch.nn.functional.normalize(text_embedding, p=2, dim=-1)
            text_embedding = text_embedding.detach().cpu().numpy()[0]
        
        # Find nearest neighbors
        top_ids, top_similarities = self.find_nearest_neighbors(
            text_embedding, self.image_embeddings, self.image_ids, k
        )
        
        # Handle case where we don't have actual images
        if not self.image_paths or len(self.image_paths) == 0:
            logger.warning("No image paths available for visualization.")
            # Just print the results
            for i, (img_id, similarity) in enumerate(zip(top_ids, top_similarities)):
                logger.info(f"Top {i+1}: Image ID {img_id}, Similarity: {similarity:.4f}")
            return
        
        # Create a figure
        fig, axes = plt.subplots(1, k, figsize=(15, 4))
        if k == 1:
            axes = [axes]  # Make iterable for single result
        
        fig.suptitle(f"Top {k} images for query: '{query_text}'", fontsize=16)
        
        # Plot each image
        for i, (ax, img_id, similarity) in enumerate(zip(axes, top_ids, top_similarities)):
            idx = self.image_ids.index(img_id)
            image_data = self.image_paths[idx]
            
            # Handle different image formats
            if isinstance(image_data, str) and (image_data.startswith('http://') or image_data.startswith('https://')):
                # URL - download image
                response = requests.get(image_data)
                img = Image.open(BytesIO(response.content))
            elif isinstance(image_data, str) and os.path.exists(image_data):
                # Local file path
                img = Image.open(image_data)
            elif isinstance(image_data, Image.Image):
                # PIL Image
                img = image_data
            elif isinstance(image_data, np.ndarray):
                # Numpy array
                img = Image.fromarray(image_data)
            elif isinstance(image_data, torch.Tensor):
                # Tensor
                img = self.tensor_to_image(image_data)
            else:
                # Unknown format
                logger.warning(f"Unsupported image format: {type(image_data)}")
                ax.text(0.5, 0.5, f"Image ID: {img_id}\nSimilarity: {similarity:.4f}",
                       ha='center', va='center')
                ax.axis('off')
                continue
            
            # Display image
            ax.imshow(img)
            ax.set_title(f"Similarity: {similarity:.4f}")
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save if filename provided
        if output_filename:
            save_path = self.output_dir / output_filename
            plt.savefig(save_path, dpi=300)
            logger.info(f"Saved visualization to {save_path}")
        
        # Display
        plt.show()
    
    def visualize_image_to_text_search(
        self,
        query_image: Union[str, Image.Image, np.ndarray, torch.Tensor],
        k: int = 5,
        output_filename: Optional[str] = None
    ) -> None:
        """Visualize top-k texts for an image query."""
        if self.text_embeddings is None or self.texts is None:
            logger.error("No text embeddings loaded. Load or generate embeddings first.")
            return
        
        logger.info("Performing image-to-text search")
        
        # Generate image embedding
        with torch.no_grad():
            # Handle different image formats
            if isinstance(query_image, str) and (query_image.startswith('http://') or query_image.startswith('https://')):
                # URL - download image
                response = requests.get(query_image)
                img = Image.open(BytesIO(response.content))
            elif isinstance(query_image, str) and os.path.exists(query_image):
                # Local file path
                img = Image.open(query_image)
            elif isinstance(query_image, Image.Image):
                # PIL Image
                img = query_image
            elif isinstance(query_image, np.ndarray):
                # Numpy array
                img = Image.fromarray(query_image)
            elif isinstance(query_image, torch.Tensor):
                # Tensor
                img = self.tensor_to_image(query_image)
            else:
                logger.error(f"Unsupported image format: {type(query_image)}")
                return
            
            # Preprocess and encode
            image_inputs = self.processor(
                images=[img],
                return_tensors="pt"
            )
            image_embedding = self.model.encode_image(image_inputs.pixel_values.to(self.device))
            image_embedding = image_embedding.to(torch.float32)
            image_embedding = torch.nn.functional.normalize(image_embedding, p=2, dim=-1)
            image_embedding = image_embedding.detach().cpu().numpy()[0]
        
        # Find nearest neighbors
        top_ids, top_similarities = self.find_nearest_neighbors(
            image_embedding, self.text_embeddings, self.text_ids, k
        )
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 2]})
        
        # Show the query image
        ax1.imshow(img)
        ax1.set_title("Query Image")
        ax1.axis('off')
        
        # Show the top text results
        ax2.axis('off')
        text_results = []
        for i, (text_id, similarity) in enumerate(zip(top_ids, top_similarities)):
            idx = self.text_ids.index(text_id)
            if idx < len(self.texts):
                text = self.texts[idx]
                text_results.append(f"{i+1}. [{similarity:.4f}] {text}")
        
        ax2.text(0, 0.5, "\n\n".join(text_results), 
                 fontsize=12, va='center', wrap=True)
        ax2.set_title(f"Top {k} Text Matches")
        
        plt.tight_layout()
        
        # Save if filename provided
        if output_filename:
            save_path = self.output_dir / output_filename
            plt.savefig(save_path, dpi=300)
            logger.info(f"Saved visualization to {save_path}")
        
        # Display
        plt.show()
    
    @staticmethod
    def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
        """Convert a tensor to a PIL Image."""
        if tensor.ndim == 4:
            tensor = tensor[0]  # Take first item from batch
        
        # Convert to numpy and adjust format
        if tensor.ndim == 3:
            # Channels are likely first (C, H, W)
            if tensor.shape[0] == 3 or tensor.shape[0] == 1:
                tensor = tensor.permute(1, 2, 0)  # Convert to (H, W, C)
        
        # Convert to uint8 for PIL
        array = tensor.detach().cpu().numpy()
        
        # Normalize to 0-255 range if needed
        if array.max() <= 1.0:
            array = (array * 255).astype(np.uint8)
        else:
            array = array.astype(np.uint8)
        
        # Create PIL image
        return Image.fromarray(array)


# Main execution
async def main():
    parser = argparse.ArgumentParser(description="CLIP Embedding Visualization Tool")
    parser.add_argument('--model', default="ViT-B/32", help="CLIP model name")
    parser.add_argument('--output-dir', default="output/visualizations", help="Output directory")
    parser.add_argument('--dataset', default="nlphuji/flickr30k", help="Dataset to use")
    parser.add_argument('--max-samples', type=int, default=500, help="Maximum samples to process")
    parser.add_argument('--save-file', default="embeddings.pkl", help="Filename to save embeddings")
    parser.add_argument('--load', action='store_true', help="Load embeddings instead of generating")
    parser.add_argument('--tsne', action='store_true', help="Generate t-SNE visualization")
    parser.add_argument('--umap', action='store_true', help="Generate UMAP visualization")
    parser.add_argument('--text-query', help="Text query for image search")
    parser.add_argument('--image-query', help="Image path for text search")
    parser.add_argument('--top-k', type=int, default=5, help="Number of top results to show")
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = EmbeddingVisualizer(
        model_name=args.model,
        output_dir=args.output_dir
    )
    
    # Load or generate embeddings
    if args.load:
        visualizer.load_embeddings(args.save_file)
    else:
        # Create dataset config
        dataset_config = DatasetConfig(
            image_datasets=[args.dataset],
            text_datasets=[args.dataset],
            batch_size=32
        )
        
        # Generate embeddings
        await visualizer.generate_and_save_embeddings(
            dataset_config,
            max_samples=args.max_samples,
            save_filename=args.save_file
        )
    
    # Generate visualizations
    if args.tsne:
        visualizer.visualize_tsne()
    
    if args.umap:
        visualizer.visualize_umap()
    
    # Perform searches
    if args.text_query:
        visualizer.visualize_text_to_image_search(
            args.text_query,
            k=args.top_k,
            output_filename=f"text_to_image_{args.text_query.replace(' ', '_')}.png"
        )
    
    if args.image_query:
        visualizer.visualize_image_to_text_search(
            args.image_query,
            k=args.top_k,
            output_filename=f"image_to_text_search.png"
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 