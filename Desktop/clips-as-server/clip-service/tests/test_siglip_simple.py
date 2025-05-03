#!/usr/bin/env python3

"""
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
Simple test for SigLIP model using open_clip.
"""

import sys
import os
import time
import logging
import torch
import numpy as np
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_siglip_with_open_clip():
    """Test SigLIP model with open_clip."""
    try:
        import open_clip
        
        logger.info("Loading SigLIP model using open_clip...")
        
        # Using ViT-B-16-SigLIP which is available in open_clip
        model_name = "ViT-B-16-SigLIP"
        pretrained = "webli"  # SigLIP models are trained on WebLI
        
        logger.info(f"Using model: {model_name} with {pretrained} weights")
        
        # Load model
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        
        # Move to CPU for testing
        device = "cpu"
        model = model.to(device)
        model.eval()
        
        logger.info(f"SigLIP model loaded successfully on {device}")
        
        # Create sample images
        logger.info("Creating sample images...")
        sample_images = [
            Image.new('RGB', (224, 224), color='red'),
            Image.new('RGB', (224, 224), color='blue')
        ]
        
        # Preprocess images
        preprocessed_images = torch.stack([preprocess(img) for img in sample_images]).to(device)
        
        # Create sample texts
        logger.info("Creating sample texts...")
        sample_texts = [
            "a red square",
            "a blue square"
        ]
        
        # Tokenize text
        text_tokens = tokenizer(sample_texts).to(device)
        
        # Get embeddings
        with torch.no_grad():
            # Get image features
            start_time = time.time()
            image_features = model.encode_image(preprocessed_images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logger.info(f"Image encoding time: {time.time() - start_time:.3f}s")
            
            # Get text features
            start_time = time.time()
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            logger.info(f"Text encoding time: {time.time() - start_time:.3f}s")
        
        # Print shapes
        logger.info(f"Image embeddings shape: {image_features.shape}")
        logger.info(f"Text embeddings shape: {text_features.shape}")
        
        # Compute similarities
        logger.info("Computing similarities...")
        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        logger.info(f"Similarity matrix:\n{similarities}")
        
        # We expect higher similarity for matching concepts (red-red, blue-blue)
        # First image (red) should be more similar to first text (red)
        # Second image (blue) should be more similar to second text (blue)
        
        # Check the diagonals have higher values
        assert similarities[0, 0] > similarities[0, 1], "Red image should be more similar to 'red square' text"
        assert similarities[1, 1] > similarities[1, 0], "Blue image should be more similar to 'blue square' text"
        
        logger.info("SigLIP test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing SigLIP model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting simple SigLIP test")
    test_siglip_with_open_clip() 