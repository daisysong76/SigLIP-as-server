"""
Test script for verifying LLaVA processor fix.
Run with: python scripts/test_llava.py --image demo_images/test_image.png
"""
import os
import sys
import argparse
import logging
import torch
import traceback
from PIL import Image, ImageDraw
from llava_utils import LLaVAProcessor, PROMPT_TEMPLATES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Known valid LLaVA models on HuggingFace
VALID_MODELS = [
    "llava-hf/llava-1.5-7b-hf",     # Primary model - should work with our fix
]

def create_test_image(image_path: str):
    """Create a test image with basic shapes for testing."""
    logger.info(f"Creating test image at {image_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    
    # Create a simple image with shapes
    img = Image.new('RGB', (224, 224), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Draw a red square
    draw.rectangle([(50, 50), (100, 100)], fill=(255, 0, 0))
    
    # Draw a blue circle
    draw.ellipse([(130, 50), (180, 100)], fill=(0, 0, 255))
    
    # Draw a green triangle
    draw.polygon([(90, 130), (140, 180), (40, 180)], fill=(0, 255, 0))
    
    # Save the image
    img.save(image_path)
    logger.info(f"Created test image with shapes at {image_path}")
    return img

def test_llava_processor(image_path, model_name=None, fallback_model=None):
    """Test the LLaVA processor with a single image."""
    logger.info(f"Testing LLaVA processor with image: {image_path}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Create a demo image if the path doesn't exist
    if not os.path.exists(image_path):
        create_test_image(image_path)
    
    # Collect models to try
    models_to_try = []
    
    # First try the specified model
    if model_name:
        models_to_try.append(model_name)
    
    # Then try the fallback model if specified
    if fallback_model and fallback_model not in models_to_try:
        models_to_try.append(fallback_model)
    
    # Finally add other valid models as additional fallbacks
    for valid_model in VALID_MODELS:
        if valid_model not in models_to_try:
            models_to_try.append(valid_model)
    
    # Try each model until one works
    success = False
    successful_model = None
    for i, model in enumerate(models_to_try):
        logger.info(f"Attempt {i+1}/{len(models_to_try)}: Testing with model {model}")
        try:
            # Initialize the processor
            logger.info(f"Initializing LLaVA processor with model: {model}")
            processor = LLaVAProcessor(model_name=model)
            
            # Test with a simple prompt
            prompt = "Describe this image with the shapes you see."
            logger.info(f"Processing image with prompt: '{prompt}'")
            
            # Run inference
            result = processor.infer(image_path, prompt)
            
            # Print the result
            logger.info("LLaVA processing successful!")
            logger.info(f"Result: {result}")
            
            success = True
            successful_model = model
            logger.info(f"Successfully tested with model: {model}")
            break
        except Exception as e:
            logger.error(f"Error with model {model}: {e}")
            logger.error(traceback.format_exc())
            logger.info(f"Failed with model {model}, {'trying next model' if i < len(models_to_try)-1 else 'all models failed'}")
    
    if success:
        logger.info(f"LLaVA processor test passed with model: {successful_model}!")
        # If we succeeded with a model different from the one specified, suggest updating the code
        if model_name and successful_model != model_name:
            logger.info(f"NOTE: The specified model '{model_name}' failed, but '{successful_model}' worked. Consider updating your code to use this model.")
        return True
    else:
        logger.error("LLaVA processor test failed with all models!")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test LLaVA processor fix")
    parser.add_argument("--image", type=str, default="demo_images/test_image.png",
                        help="Path to test image")
    parser.add_argument("--model", type=str, default="llava-hf/llava-1.5-7b-hf",
                        help="LLaVA model name")
    parser.add_argument("--fallback-model", type=str, default=None,
                        help="Fallback model if primary fails")
    parser.add_argument("--force-recreate", action="store_true", 
                        help="Force recreation of test image even if it exists")
    args = parser.parse_args()
    
    # Create or recreate test image if needed
    if args.force_recreate or not os.path.exists(args.image):
        create_test_image(args.image)
    
    success = test_llava_processor(args.image, args.model, args.fallback_model)
    
    if success:
        logger.info("LLaVA processor test completed successfully!")
        return 0
    else:
        logger.error("LLaVA processor test failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 