#!/usr/bin/env python3
"""
Test for the updated SigLIP model implementation.
test_siglip_simple.py
Purpose: Basic proof-of-concept test for SigLIP
Features:
Directly imports open_clip
Uses simple colored squares (red and blue) for testing
Minimal functionality
No command-line arguments
Tests on CPU only
No metrics calculation beyond basic similarity check
No visualization
Does not use the custom model wrapper
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the SigLIP model
from model.siglip_model import load_siglip_model

def main():
    """Test the SigLIP model with simple colored squares."""
    print("Testing updated SigLIP model implementation...")
    
    # Load the model
    model = load_siglip_model(
        model_name="ViT-B-16-SigLIP",
        pretrained="webli",
        device=None,  # Auto-select device
        initial_batch_size=2,
        max_batch_size=4,
        cache_embeddings=True
    )
    
    print(f"Model loaded successfully: {model.model_name} with {model.pretrained} weights on {model.device}")
    
    # Create test images - red and blue squares
    red_image = Image.new('RGB', (224, 224), color='red')
    blue_image = Image.new('RGB', (224, 224), color='blue')
    
    # Create test texts
    texts = ["a red square", "a blue square"]
    
    # Encode images
    image_outputs = model.encode_images([red_image, blue_image])
    image_embeddings = image_outputs["image_embeddings"]
    
    print(f"Images encoded successfully. Shape: {image_embeddings.shape}")
    
    # Encode texts
    text_outputs = model.encode_text(texts)
    text_embeddings = text_outputs["text_embeddings"]
    
    print(f"Texts encoded successfully. Shape: {text_embeddings.shape}")
    
    # Compute similarity matrix
    similarity = model.compute_similarity(image_embeddings, text_embeddings)
    
    print("\nSimilarity matrix:")
    print(f"{similarity.numpy()}")
    
    # Check if the similarity matrix makes sense (red with "red", blue with "blue")
    if similarity[0, 0] > similarity[0, 1] and similarity[1, 1] > similarity[1, 0]:
        print("\n✅ Test PASSED! Similarity matrix shows correct correlations.")
        print(f"Red image ↔ 'a red square': {similarity[0, 0]:.4f}")
        print(f"Blue image ↔ 'a blue square': {similarity[1, 1]:.4f}")
        print(f"Red image ↔ 'a blue square': {similarity[0, 1]:.4f}")
        print(f"Blue image ↔ 'a red square': {similarity[1, 0]:.4f}")
    else:
        print("\n❌ Test FAILED! Similarity matrix doesn't show expected correlations.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 