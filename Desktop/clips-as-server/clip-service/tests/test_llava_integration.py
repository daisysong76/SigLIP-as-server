#!/usr/bin/env python3
"""
Simple test script to verify image and text preprocessing for CLIP and LLaVA models.
"""

import sys
import os
import logging
import torch
from PIL import Image
from typing import List, Union
import ast

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_images(image_inputs: List[Union[str, Image.Image]]) -> List[Image.Image]:
    """
    Convert image paths or mixed inputs to PIL images.
    
    Args:
        image_inputs: List of image paths or PIL images
        
    Returns:
        List of PIL images
    """
    processed_images = []
    
    for img in image_inputs:
        if isinstance(img, str):
            # This is a path - in production code, add error handling
            try:
                # In real code, you'd check if the file exists
                # For demonstration, we'll create a dummy image
                dummy_image = Image.new('RGB', (224, 224), color='blue')
                processed_images.append(dummy_image)
                logger.info(f"Converted path to PIL Image: {img}")
            except Exception as e:
                logger.error(f"Error loading image {img}: {e}")
                # Create a black image as placeholder
                dummy_image = Image.new('RGB', (224, 224), color='black')
                processed_images.append(dummy_image)
        elif isinstance(img, Image.Image):
            # Already a PIL image
            processed_images.append(img)
        else:
            logger.warning(f"Unsupported image type: {type(img)}")
            # Create a placeholder
            dummy_image = Image.new('RGB', (224, 224), color='red')
            processed_images.append(dummy_image)
    
    return processed_images

def prepare_texts(text_inputs):
    """
    Properly handle various text input formats.
    
    Args:
        text_inputs: Text inputs in various formats
        
    Returns:
        List of string captions
    """
    processed_texts = []
    
    for text in text_inputs:
        if isinstance(text, list):
            # It's already a list - make sure all elements are strings
            processed_texts.extend([str(t) for t in text])
            logger.info(f"Added {len(text)} captions from list")
        elif isinstance(text, str):
            # Check if it's a stringified list
            if text.startswith('[') and text.endswith(']'):
                try:
                    # Safely parse the string as a list
                    parsed_list = ast.literal_eval(text)
                    if isinstance(parsed_list, list):
                        processed_texts.extend([str(t) for t in parsed_list])
                        logger.info(f"Parsed string list into {len(parsed_list)} captions")
                        continue
                except (ValueError, SyntaxError):
                    # Not a valid list representation, treat as a single string
                    pass
            
            # Just a normal string
            processed_texts.append(text)
            logger.info(f"Added single caption: {text[:20]}...")
        else:
            logger.warning(f"Unexpected text type: {type(text)}")
            processed_texts.append(str(text))
    
    return processed_texts

def test_preprocessing():
    """Test preprocessing functions for images and text."""
    try:
        # Import necessary processors (if available)
        try:
            from transformers import CLIPProcessor
            processor_available = True
        except ImportError:
            logger.warning("CLIPProcessor not available, skipping tensor conversion")
            processor_available = False

        # 1. Test image preprocessing
        logger.info("Testing image preprocessing...")
        
        # Create sample problematic image inputs
        sample_image_inputs = [
            "path/to/image1.jpg",  # String path
            "path/to/image2.jpg",
            Image.new('RGB', (224, 224), color='green'),  # Actual PIL image
            ["nested", "list", "of", "paths"]  # Problematic nested structure
        ]
        
        # Fix image inputs
        fixed_images = []
        for item in sample_image_inputs:
            if isinstance(item, list):
                # Handle nested list
                fixed_images.extend(prepare_images(item))
            else:
                fixed_images.append(item)
                
        pil_images = prepare_images(fixed_images)
        logger.info(f"Successfully processed {len(pil_images)} images")
        
        # Process with CLIP processor if available
        if processor_available:
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            inputs = processor(images=pil_images, return_tensors="pt")
            logger.info(f"Image tensor shape: {inputs.pixel_values.shape}")
        
        # 2. Test text preprocessing
        logger.info("\nTesting text preprocessing...")
        
        # Create sample problematic text inputs
        sample_text_inputs = [
            ["Caption 1", "Caption 2"],  # Regular list
            '["Stringified list caption 1", "Stringified list caption 2"]',  # Problematic stringified list
            "Single caption"  # Single caption
        ]
        
        # Process texts
        processed_texts = prepare_texts(sample_text_inputs)
        logger.info(f"Successfully processed {len(processed_texts)} text items")
        
        # Process with CLIP processor if available
        if processor_available:
            text_inputs = processor(text=processed_texts, return_tensors="pt", padding=True)
            logger.info(f"Text tensor shape: {text_inputs.input_ids.shape}")
        
        logger.info("All preprocessing tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Error in preprocessing tests: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting preprocessing tests")
    test_preprocessing()
    logger.info("Tests completed") 