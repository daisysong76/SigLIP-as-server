#!/usr/bin/env python3
"""
Qwen-VL utilities for multimodal reasoning.
This module provides a wrapper around the Qwen-VL model for visual reasoning tasks,
designed as a drop-in replacement for LLaVA in the embedding pipeline.
"""

import os
import logging
from typing import Dict, Optional, Union, List
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define prompt templates similar to LLaVA
PROMPT_TEMPLATES = {
    "caption": "Please provide a detailed caption for this image.",
    "scene": "Describe the scene in this image in detail. What are the key elements, setting, and mood?",
    "table": "Extract structured information from this image and format it as a table or list.",
    "qa": "What objects can you see in this image? Are there any people? What activities are happening?",
    "explanation": "Analyze this image and explain its significance, context, and possible implications."
}

class QwenVLProcessor:
    """
    A processor for running Qwen-VL model inference on images with text prompts.
    """
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen-VL-Chat", 
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the Qwen-VL processor.
        
        Args:
            model_name: Name of the Qwen-VL model to use
            device: Device to run the model on ("cuda" or "cpu")
        """
        logger.info(f"Initializing Qwen-VL processor with model: {model_name} on {device}")
        
        self.model_name = model_name
        self.device = device
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                trust_remote_code=True
            ).eval()
            logger.info(f"Successfully loaded Qwen-VL model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading Qwen-VL model: {e}")
            raise
    
    def infer(self, image: Union[Image.Image, str], prompt: str) -> str:
        """
        Run inference on an image with a text prompt.
        
        Args:
            image: PIL.Image object or path to image
            prompt: Text prompt to guide the model's response
            
        Returns:
            Model's response as a string
        """
        try:
            # Handle image input (file path or PIL Image)
            if isinstance(image, str):
                if os.path.exists(image):
                    image = Image.open(image).convert('RGB')
                else:
                    raise FileNotFoundError(f"Image file not found: {image}")
            
            # Format the prompt according to Qwen-VL's expected chat format
            query = self.tokenizer.from_list_format([
                {"image": image},
                {"text": prompt}
            ])
            
            # Generate response
            with torch.inference_mode():
                response, _ = self.model.chat(self.tokenizer, query=query, history=None)
            
            logger.debug(f"Prompt: {prompt}")
            logger.debug(f"Response: {response}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error during Qwen-VL inference: {e}")
            return f"Error during processing: {str(e)}"
    
    def batch_process(self, images: List[Union[Image.Image, str]], prompts: List[str]) -> List[str]:
        """
        Process multiple images with their corresponding prompts.
        
        Args:
            images: List of PIL.Image objects or paths to images
            prompts: List of text prompts
            
        Returns:
            List of model responses
        """
        if len(images) != len(prompts):
            raise ValueError("Number of images must match number of prompts")
        
        results = []
        for i, (image, prompt) in enumerate(zip(images, prompts)):
            logger.info(f"Processing image {i+1}/{len(images)}")
            result = self.infer(image, prompt)
            results.append(result)
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = QwenVLProcessor()
    
    # Test with a sample image
    image_path = "test_images/sample.jpg"
    if os.path.exists(image_path):
        result = processor.infer(image_path, PROMPT_TEMPLATES["caption"])
        print(f"Caption: {result}")
    else:
        print(f"Test image not found: {image_path}") 