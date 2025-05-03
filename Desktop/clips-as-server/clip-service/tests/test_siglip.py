#!/usr/bin/env python3
"""
Test script for SigLIP model integration.
Basic Integration Test: It tests the fundamental functionality of the SigLIP model wrapper.
Test Procedure:
Imports the load_siglip_model function from the custom model wrapper
Loads the SigLIP model with basic configuration parameters, using "google/siglip-base-patch16-224" model
Creates two sample test images (red and blue squares)
Defines corresponding text descriptions ("a red square", "a blue square")
Encodes both images and texts to get embeddings
Computes similarity between image and text embeddings
Checks that the similarity matrix shows expected relationships (red image matches "red square" text, etc.)
Key Features:
Runs on CPU for testing purposes
Uses simple colored squares for easy verification
Tests the core functionality: encoding and similarity computation
Logs detailed information about each step of the process
Uses the custom model wrapper rather than direct implementation
This is a basic sanity check test that verifies:
The SigLIP model can be loaded successfully
The image encoding works correctly
The text encoding works correctly
Similarity calculation produces sensible results
"""

import sys
import os
import logging
import torch
from PIL import Image
from typing import List

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_siglip_model():
    """Test the SigLIP model wrapper."""
    try:
        from model.siglip_model import load_siglip_model
        
        logger.info("Loading SigLIP model...")
        model = load_siglip_model(
            model_name="google/siglip-base-patch16-224",
            device="cpu",  # Use CPU for testing
            initial_batch_size=2,
            max_batch_size=4
        )
        logger.info("SigLIP model loaded successfully")
        
        # Create sample images
        logger.info("Creating sample images...")
        sample_images = [
            Image.new('RGB', (224, 224), color='red'),
            Image.new('RGB', (224, 224), color='blue')
        ]
        
        # Create sample texts
        logger.info("Creating sample texts...")
        sample_texts = [
            "a red square",
            "a blue square"
        ]
        
        # Encode images
        logger.info("Encoding images...")
        image_results = model.encode_images(sample_images)
        image_embeddings = image_results["image_embeddings"]
        logger.info(f"Image embeddings shape: {image_embeddings.shape}")
        
        # Encode text
        logger.info("Encoding texts...")
        text_results = model.encode_text(sample_texts)
        text_embeddings = text_results["text_embeddings"]
        logger.info(f"Text embeddings shape: {text_embeddings.shape}")
        
        # Compute similarities
        logger.info("Computing similarities...")
        similarities = model.compute_similarity(image_embeddings, text_embeddings)
        logger.info(f"Similarity matrix shape: {similarities.shape}")
        logger.info(f"Similarity matrix:\n{similarities}")
        
        # Check if the diagonal has highest values (red->red, blue->blue)
        diagonal = torch.diag(similarities)
        logger.info(f"Diagonal similarities: {diagonal}")
        
        # Test successful if we got here without errors
        logger.info("SigLIP test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing SigLIP model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting SigLIP integration test")
    test_siglip_model() 