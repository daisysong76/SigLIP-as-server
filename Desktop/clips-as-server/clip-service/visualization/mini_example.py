#!/usr/bin/env python3
"""
Minimal working example for CLIP embedding visualization
with proper text and image processing.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import clip
from transformers import CLIPProcessor
from datasets import load_dataset
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pickle
from pathlib import Path
import logging
import asyncio
from typing import List, Dict, Any
import matplotlib.colors as mcolors

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleVisualizer:
    def __init__(self, output_dir="output/mini_example"):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and processor
        logger.info("Loading CLIP model...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", 
            torch_dtype=torch.float32
        )
        
        # Storage for embeddings
        self.image_embeddings = None
        self.text_embeddings = None
        self.image_ids = []
        self.text_ids = []
        self.texts = []
        self.class_names = {}
    
    async def generate_embeddings(self, dataset_name="cifar100", max_samples=100):
        """Generate embeddings from a dataset with proper processing."""
        logger.info(f"Loading dataset: {dataset_name}")
        
        # Load dataset
        try:
            ds = load_dataset(dataset_name, split="train", streaming=True)
            logger.info(f"Dataset loaded. Column names: {ds.column_names}")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return False
        
        # Create text descriptions for classes (for CIFAR-100)
        if dataset_name == "cifar100":
            # First get the class names
            class_names = {}
            try:
                class_info = load_dataset(dataset_name, split="train")
                if "fine_label" in class_info.features:
                    class_names = {i: name for i, name in enumerate(class_info.features["fine_label"].names)}
                    logger.info(f"Found {len(class_names)} class names")
                else:
                    logger.warning("Could not find class names")
            except Exception as e:
                logger.error(f"Error loading class names: {e}")
                # Use numbers as class names as fallback
                class_names = {i: f"class {i}" for i in range(100)}
        
        # Prepare for collecting embeddings
        image_embeddings = []
        text_embeddings = []
        
        # For text embeddings, create simple descriptions
        if dataset_name == "cifar100":
            # Create text descriptions for each class
            raw_texts = [f"a photo of a {name}" for _, name in class_names.items()]
            text_ids = list(range(len(raw_texts)))
            
            # Process in small batches
            batch_size = 16
            for i in range(0, len(raw_texts), batch_size):
                batch_texts = raw_texts[i:i+batch_size]
                
                # Process with CLIP
                with torch.no_grad():
                    # CRITICAL FIX: Explicitly use tokenize from clip and set max_length
                    tokens = clip.tokenize(batch_texts, truncate=True).to(self.device)
                    
                    # Generate embeddings
                    features = self.model.encode_text(tokens)
                    features = features.to(torch.float32)
                    features = torch.nn.functional.normalize(features, p=2, dim=1)
                    text_embeddings.append(features.detach().cpu().numpy())
            
            # Save the raw texts
            self.texts = raw_texts
            self.text_ids = text_ids
            
            # Combine all batches
            self.text_embeddings = np.concatenate(text_embeddings, axis=0)
            logger.info(f"Generated text embeddings: {self.text_embeddings.shape}")
        
        # For image embeddings with CIFAR-100
        if dataset_name == "cifar100":
            try:
                # Load non-streaming dataset with specific number of samples
                logger.info(f"Loading non-streaming CIFAR-100 dataset with up to {max_samples} samples")
                cifar_ds = load_dataset("cifar100", split=f"train[:{max_samples}]")
                logger.info(f"Loaded dataset with {len(cifar_ds)} samples")
                
                # Debug first item
                sample = cifar_ds[0]
                logger.info(f"Sample keys: {list(sample.keys())}")
                logger.info(f"Sample fine_label: {sample['fine_label']}")
                logger.info(f"Image type: {type(sample['img'])}")
                
                # Process images in batches
                batch_size = 16
                count = 0
                image_ids = []
                
                # Loop over the dataset in batches
                for i in range(0, len(cifar_ds), batch_size):
                    end_idx = min(i + batch_size, len(cifar_ds))
                    logger.info(f"Processing batch {i}:{end_idx}")
                    
                    # Get batch of samples directly from the dataset
                    batch_images = []
                    batch_labels = []
                    
                    # Process each item in the batch
                    for j in range(i, end_idx):
                        try:
                            # Get sample
                            sample = cifar_ds[j]
                            
                            # Get image and label
                            img = sample['img']  # This is a PIL Image
                            label = sample['fine_label']  # This is the class label
                            
                            # Process image with CLIP preprocessing
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            
                            # Preprocess the image
                            processed_img = self.preprocess(img)
                            
                            # Add to batch
                            batch_images.append(processed_img)
                            batch_labels.append(label)
                            
                            # Count processed items
                            count += 1
                        except Exception as e:
                            logger.error(f"Error processing sample {j}: {str(e)}")
                    
                    # Only continue if we have images in the batch
                    if batch_images:
                        try:
                            # Convert list of tensors to a batch tensor
                            image_batch = torch.stack(batch_images)
                            logger.info(f"Created image batch with shape: {image_batch.shape}")
                            
                            # Generate embeddings
                            with torch.no_grad():
                                image_batch = image_batch.to(self.device)
                                features = self.model.encode_image(image_batch)
                                features = features.to(torch.float32)
                                features = torch.nn.functional.normalize(features, p=2, dim=1)
                                
                                # Add to list of embeddings
                                image_embeddings.append(features.detach().cpu().numpy())
                            
                            # Add labels to list of IDs
                            image_ids.extend(batch_labels)
                            
                            logger.info(f"Processed {count} images so far")
                        except Exception as e:
                            logger.error(f"Error generating embeddings for batch: {str(e)}")
                
                # Save the image IDs
                self.image_ids = image_ids
                
                # Combine all batches of embeddings
                if image_embeddings:
                    self.image_embeddings = np.concatenate(image_embeddings, axis=0)
                    logger.info(f"Generated image embeddings with shape: {self.image_embeddings.shape}")
                else:
                    logger.error("No image embeddings were generated")
                    return False
            except Exception as e:
                logger.error(f"Error processing CIFAR-100 images: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return False
        
        # Save embeddings
        save_path = self.output_dir / "mini_embeddings.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump({
                'image_embeddings': self.image_embeddings,
                'text_embeddings': self.text_embeddings,
                'image_ids': self.image_ids,
                'text_ids': self.text_ids,
                'texts': self.texts
            }, f)
        
        logger.info(f"Saved embeddings to {save_path}")
        
        # Return success
        return True
    
    def visualize_tsne(self, colored_by_class=True):
        """Visualize embeddings using t-SNE with enhanced coloring."""
        if self.image_embeddings is None or self.text_embeddings is None:
            logger.error("No embeddings found. Generate embeddings first.")
            return
        
        logger.info(f"Visualizing embeddings with t-SNE... Image shape: {self.image_embeddings.shape}, Text shape: {self.text_embeddings.shape}")
        
        # Combine embeddings for joint visualization
        combined_embeddings = np.vstack([self.image_embeddings, self.text_embeddings])
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
        embeddings_2d = tsne.fit_transform(combined_embeddings)
        
        # Split back into image and text embeddings
        image_embeddings_2d = embeddings_2d[:len(self.image_embeddings)]
        text_embeddings_2d = embeddings_2d[len(self.image_embeddings):]
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Color by image/text type (default)
        if not colored_by_class:
            plt.scatter(
                image_embeddings_2d[:, 0], image_embeddings_2d[:, 1],
                c='blue', label='Images', alpha=0.5, s=5
            )
            plt.scatter(
                text_embeddings_2d[:, 0], text_embeddings_2d[:, 1],
                c='red', label='Text', alpha=0.5, s=5
            )
            plt.title('t-SNE Visualization of CLIP Embeddings')
        else:
            # Color by class label (when available)
            unique_labels = set(self.image_ids)
            cmap = plt.cm.get_cmap('tab20', len(unique_labels))
            
            # Create a colormap for the classes
            colors = {label: cmap(i) for i, label in enumerate(unique_labels)}
            
            # Plot each class separately
            for label in unique_labels:
                # Get indices for this class
                img_indices = [i for i, id_val in enumerate(self.image_ids) if id_val == label]
                txt_indices = [i for i, id_val in enumerate(self.text_ids) if id_val == label]
                
                # Get class name if available
                class_name = self.class_names.get(label, f"Class {label}")
                
                # Plot images for this class
                if img_indices:
                    plt.scatter(
                        image_embeddings_2d[img_indices, 0], 
                        image_embeddings_2d[img_indices, 1],
                        c=[colors[label]], 
                        marker='o',
                        alpha=0.6, 
                        s=30,
                        label=f"Image: {class_name}"
                    )
                
                # Plot text for this class  
                if txt_indices:
                    plt.scatter(
                        text_embeddings_2d[txt_indices, 0], 
                        text_embeddings_2d[txt_indices, 1],
                        c=[colors[label]], 
                        marker='x',
                        alpha=0.8, 
                        s=50,
                        label=f"Text: {class_name}"
                    )
                    
                    # Connect matching image-text pairs with lines
                    for img_idx in img_indices:
                        for txt_idx in txt_indices:
                            plt.plot(
                                [image_embeddings_2d[img_idx, 0], text_embeddings_2d[txt_idx, 0]],
                                [image_embeddings_2d[img_idx, 1], text_embeddings_2d[txt_idx, 1]],
                                c=colors[label], alpha=0.15, linewidth=0.5
                            )
            
            plt.title('t-SNE Visualization of CLIP Embeddings (Colored by Class)')
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.tight_layout()
        
        # Save plot
        filename = "tsne_visualization_by_class.png" if colored_by_class else "tsne_visualization.png"
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Saved t-SNE visualization to {save_path}")
    
    def visualize_pca(self, colored_by_class=True):
        """Visualize embeddings using PCA with enhanced coloring."""
        if self.image_embeddings is None or self.text_embeddings is None:
            logger.error("No embeddings found. Generate embeddings first.")
            return
        
        logger.info(f"Visualizing embeddings with PCA... Image shape: {self.image_embeddings.shape}, Text shape: {self.text_embeddings.shape}")
        
        # Combine embeddings for joint visualization
        combined_embeddings = np.vstack([self.image_embeddings, self.text_embeddings])
        
        # Apply PCA
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(combined_embeddings)
        
        # Get explained variance
        explained_variance = pca.explained_variance_ratio_
        
        # Split back into image and text embeddings
        image_embeddings_2d = embeddings_2d[:len(self.image_embeddings)]
        text_embeddings_2d = embeddings_2d[len(self.image_embeddings):]
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Color by image/text type (default)
        if not colored_by_class:
            plt.scatter(
                image_embeddings_2d[:, 0], image_embeddings_2d[:, 1],
                c='blue', label='Images', alpha=0.5, s=5
            )
            plt.scatter(
                text_embeddings_2d[:, 0], text_embeddings_2d[:, 1],
                c='red', label='Text', alpha=0.5, s=5
            )
            plt.title('PCA Visualization of CLIP Embeddings')
        else:
            # Color by class label (when available)
            unique_labels = set(self.image_ids)
            cmap = plt.cm.get_cmap('tab20', len(unique_labels))
            
            # Create a colormap for the classes
            colors = {label: cmap(i) for i, label in enumerate(unique_labels)}
            
            # Plot each class separately
            for label in unique_labels:
                # Get indices for this class
                img_indices = [i for i, id_val in enumerate(self.image_ids) if id_val == label]
                txt_indices = [i for i, id_val in enumerate(self.text_ids) if id_val == label]
                
                # Get class name if available
                class_name = self.class_names.get(label, f"Class {label}")
                
                # Plot images for this class
                if img_indices:
                    plt.scatter(
                        image_embeddings_2d[img_indices, 0], 
                        image_embeddings_2d[img_indices, 1],
                        c=[colors[label]], 
                        marker='o',
                        alpha=0.6, 
                        s=30,
                        label=f"Image: {class_name}"
                    )
                
                # Plot text for this class  
                if txt_indices:
                    plt.scatter(
                        text_embeddings_2d[txt_indices, 0], 
                        text_embeddings_2d[txt_indices, 1],
                        c=[colors[label]], 
                        marker='x',
                        alpha=0.8, 
                        s=50,
                        label=f"Text: {class_name}"
                    )
                    
                    # Connect matching image-text pairs with lines
                    for img_idx in img_indices:
                        for txt_idx in txt_indices:
                            plt.plot(
                                [image_embeddings_2d[img_idx, 0], text_embeddings_2d[txt_idx, 0]],
                                [image_embeddings_2d[img_idx, 1], text_embeddings_2d[txt_idx, 1]],
                                c=colors[label], alpha=0.15, linewidth=0.5
                            )
            
            plt.title('PCA Visualization of CLIP Embeddings (Colored by Class)')
        
        # Add explained variance to axis labels
        plt.xlabel(f'PC1 ({explained_variance[0]:.2%} explained variance)')
        plt.ylabel(f'PC2 ({explained_variance[1]:.2%} explained variance)')
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.tight_layout()
        
        # Save plot
        filename = "pca_visualization_by_class.png" if colored_by_class else "pca_visualization.png"
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Saved PCA visualization to {save_path}")
        
    def compare_pca_tsne(self):
        """Compare PCA and t-SNE visualizations side by side."""
        if self.image_embeddings is None or self.text_embeddings is None:
            logger.error("No embeddings found. Generate embeddings first.")
            return
        
        logger.info("Comparing PCA and t-SNE visualizations...")
        
        # Combine embeddings for joint visualization
        combined_embeddings = np.vstack([self.image_embeddings, self.text_embeddings])
        
        # Apply PCA
        pca = PCA(n_components=2, random_state=42)
        pca_embeddings = pca.fit_transform(combined_embeddings)
        explained_variance = pca.explained_variance_ratio_
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
        tsne_embeddings = tsne.fit_transform(combined_embeddings)
        
        # Split back into image and text embeddings
        pca_image = pca_embeddings[:len(self.image_embeddings)]
        pca_text = pca_embeddings[len(self.image_embeddings):]
        tsne_image = tsne_embeddings[:len(self.image_embeddings)]
        tsne_text = tsne_embeddings[len(self.image_embeddings):]
        
        # Create a 1x2 subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Color by class label
        unique_labels = set(self.image_ids)
        cmap = plt.cm.get_cmap('tab20', len(unique_labels))
        colors = {label: cmap(i) for i, label in enumerate(unique_labels)}
        
        # Plot PCA
        for label in unique_labels:
            img_indices = [i for i, id_val in enumerate(self.image_ids) if id_val == label]
            txt_indices = [i for i, id_val in enumerate(self.text_ids) if id_val == label]
            class_name = self.class_names.get(label, f"Class {label}")
            
            if img_indices:
                ax1.scatter(
                    pca_image[img_indices, 0], pca_image[img_indices, 1],
                    c=[colors[label]], marker='o', alpha=0.6, s=30,
                    label=f"Image: {class_name}" if label == list(unique_labels)[0] else ""
                )
            
            if txt_indices:
                ax1.scatter(
                    pca_text[txt_indices, 0], pca_text[txt_indices, 1],
                    c=[colors[label]], marker='x', alpha=0.8, s=50,
                    label=f"Text: {class_name}" if label == list(unique_labels)[0] else ""
                )
        
        ax1.set_title('PCA Visualization')
        ax1.set_xlabel(f'PC1 ({explained_variance[0]:.2%} explained variance)')
        ax1.set_ylabel(f'PC2 ({explained_variance[1]:.2%} explained variance)')
        
        # Plot t-SNE
        for label in unique_labels:
            img_indices = [i for i, id_val in enumerate(self.image_ids) if id_val == label]
            txt_indices = [i for i, id_val in enumerate(self.text_ids) if id_val == label]
            class_name = self.class_names.get(label, f"Class {label}")
            
            if img_indices:
                ax2.scatter(
                    tsne_image[img_indices, 0], tsne_image[img_indices, 1],
                    c=[colors[label]], marker='o', alpha=0.6, s=30
                )
            
            if txt_indices:
                ax2.scatter(
                    tsne_text[txt_indices, 0], tsne_text[txt_indices, 1],
                    c=[colors[label]], marker='x', alpha=0.8, s=50
                )
                
                # Connect matching pairs in t-SNE visualization
                if len(img_indices) > 0 and len(txt_indices) > 0:
                    img_idx = img_indices[0]
                    txt_idx = txt_indices[0]
                    ax2.plot(
                        [tsne_image[img_idx, 0], tsne_text[txt_idx, 0]],
                        [tsne_image[img_idx, 1], tsne_text[txt_idx, 1]],
                        c=colors[label], alpha=0.4, linewidth=0.8
                    )
        
        ax2.set_title('t-SNE Visualization')
        
        # Add legends with class colors
        legend_elements = []
        for i, label in enumerate(sorted(unique_labels)):
            if i < 10:  # Limit to 10 classes to avoid cluttering
                class_name = self.class_names.get(label, f"Class {label}")
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                              markerfacecolor=colors[label], markersize=10,
                                              label=class_name))
                
        # Add common legend
        fig.legend(handles=[
            plt.Line2D([0], [0], marker='o', color='w', label='Image', markerfacecolor='gray', markersize=10),
            plt.Line2D([0], [0], marker='x', color='gray', label='Text', markersize=10)
        ] + legend_elements, 
        loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=4)
        
        plt.tight_layout()
        
        # Save comparison
        save_path = self.output_dir / "pca_vs_tsne_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Saved comparison visualization to {save_path}")

    def load_embeddings(self, filename="mini_embeddings.pkl"):
        """Load embeddings from file."""
        file_path = self.output_dir / filename
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            self.image_embeddings = data.get('image_embeddings')
            self.text_embeddings = data.get('text_embeddings')
            self.image_ids = data.get('image_ids', [])
            self.text_ids = data.get('text_ids', [])
            self.texts = data.get('texts', [])
            
            # Check if we have valid embeddings
            if self.image_embeddings is None or self.text_embeddings is None:
                logger.error("Invalid embedding data in file (missing embeddings)")
                return False
                
            image_shape = "None" if self.image_embeddings is None else self.image_embeddings.shape
            text_shape = "None" if self.text_embeddings is None else self.text_embeddings.shape
            logger.info(f"Loaded embeddings: Images {image_shape}, Texts {text_shape}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return False

async def main():
    visualizer = SimpleVisualizer()
    success = False
    
    # Check if embeddings already exist
    if not visualizer.load_embeddings():
        # Generate new embeddings
        success = await visualizer.generate_embeddings(dataset_name="cifar100", max_samples=100)
    else:
        success = True
    
    # Visualize embeddings if we have them
    if success:
        # Load class names for CIFAR-100
        try:
            class_info = load_dataset("cifar100", split="train")
            if "fine_label" in class_info.features:
                visualizer.class_names = {i: name for i, name in enumerate(class_info.features["fine_label"].names)}
                logger.info(f"Loaded {len(visualizer.class_names)} class names for visualization")
        except Exception as e:
            logger.error(f"Could not load class names: {e}")
        
        # Generate various visualizations
        visualizer.visualize_tsne(colored_by_class=True)
        visualizer.visualize_pca(colored_by_class=True)
        visualizer.compare_pca_tsne()
    else:
        logger.error("Failed to load or generate embeddings. Cannot visualize.")

if __name__ == "__main__":
    asyncio.run(main()) 