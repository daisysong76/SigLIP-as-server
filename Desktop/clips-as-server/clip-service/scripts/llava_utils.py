"""
LLaVA utilities module for clean, modular visual reasoning.
Provides functions for model loading, inference, and batch processing.
This is the core LLaVA reasoning utility module
Provides a clean, modular LLaVAProcessor class for visual reasoning
Handles model loading, inference, and batch processing
Contains predefined prompt templates for different reasoning tasks (caption, scene description, table analysis, etc.)
Fundamental building block for LLaVA reasoning

"""
import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPImageProcessor,
    LlavaForConditionalGeneration,
    LlavaProcessor
)
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Any, Union, Optional
import os
import logging

# Configure logger
logger = logging.getLogger(__name__)

# Standard prompt templates
PROMPT_TEMPLATES = {
    "caption": "Describe this image.",
    "scene": "What is happening in this image? Describe the scene in detail.",
    "table": "If there is a table or figure in this image, summarize its content.",
    "qa": "What is the main object in this image? What is unusual or interesting here?",
    "explanation": "Explain the key elements and relationships in this image."
}

class LLaVAProcessor:
    """Encapsulates LLaVA model loading and inference in a clean, reusable class."""
    
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf", device: Optional[str] = None):
        """Initialize LLaVA model and processor.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cuda', 'cpu', etc.). If None, will auto-detect.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        
        # Try multiple approaches to load the model
        try:
            logger.info(f"Loading LLaVA model: {model_name}")
            
            # First try the recommended approach for llava-hf models
            if "llava-hf" in model_name.lower():
                try:
                    # Use proper model loading that includes the image_processor config
                    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                    model = LlavaForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                        trust_remote_code=True
                    ).to(device)
                    
                    # Store both elements
                    self.processor = processor
                    self.model = model
                    logger.info("Successfully loaded with AutoProcessor and LlavaForConditionalGeneration")
                except Exception as e:
                    logger.warning(f"Error loading with AutoProcessor: {e}")
                    raise
            else:
                # For other models, try fallback approaches
                try:
                    self.processor = LlavaProcessor.from_pretrained(
                        model_name,
                        use_fast=False,
                        trust_remote_code=True
                    )
                    self.model = LlavaForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                        trust_remote_code=True
                    ).to(device)
                    logger.info("Successfully loaded with LlavaProcessor and LlavaForConditionalGeneration")
                except (OSError, ValueError) as e:
                    logger.warning(f"Failed to load with LlavaProcessor: {e}, trying alternative methods")
                    
                    # Try loading with components separately
                    try:
                        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                            trust_remote_code=True
                        ).to(device)
                        
                        # Try to load tokenizer and image processor separately
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            model_name,
                            trust_remote_code=True
                        )
                        
                        # Try to find CLIP image processor for this model
                        try:
                            self.image_processor = CLIPImageProcessor.from_pretrained(
                                "openai/clip-vit-large-patch14"
                            )
                        except Exception as img_err:
                            logger.warning(f"Error loading CLIP image processor: {img_err}")
                            # Fallback to a simpler image processor
                            self.image_processor = CLIPImageProcessor(
                                size={"height": 224, "width": 224},
                                crop_size={"height": 224, "width": 224},
                                do_resize=True,
                                do_normalize=True
                            )
                        
                        # Create a combined processor
                        self.processor = {
                            "tokenizer": self.tokenizer,
                            "image_processor": self.image_processor
                        }
                        logger.info("Successfully loaded model with separate components")
                    except Exception as comp_err:
                        logger.error(f"Error loading with components: {comp_err}")
                        raise
                    
            self.model.eval()  # Set to evaluation mode
            logger.info(f"Successfully loaded model on {device}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image: Union[Image.Image, str]) -> Image.Image:
        """Preprocess image to ensure correct format for LLaVA."""
        if isinstance(image, str):
            try:
                image = Image.open(image).convert("RGB")
                logger.info(f"Loaded image with size {image.size}")
            except Exception as e:
                logger.error(f"Error loading image from {image}: {e}")
                raise
        
        # Resize if needed - use a default size if we can't detect the expected size
        target_size = 224  # Default size
        
        # Check if we have an image processor and try to get its size
        if isinstance(self.processor, dict) and "image_processor" in self.processor:
            # For component-based processor
            if hasattr(self.processor["image_processor"], "size"):
                size_dict = self.processor["image_processor"].size
                if isinstance(size_dict, dict) and "height" in size_dict:
                    target_size = size_dict["height"]
        elif hasattr(self.processor, "image_processor"):
            # For LlavaProcessor
            if hasattr(self.processor.image_processor, "size"):
                if isinstance(self.processor.image_processor.size, dict) and "height" in self.processor.image_processor.size:
                    target_size = self.processor.image_processor.size["height"]
                
        # Apply resize if needed
        if image.width != target_size or image.height != target_size:
            logger.info(f"Resizing image from {image.size} to {target_size}x{target_size}")
            image = image.resize((target_size, target_size))
        
        return image
    
    def infer(self, image: Union[Image.Image, str], prompt: str, max_new_tokens: int = 128) -> str:
        """Run LLaVA inference on a single image.
        
        Args:
            image: PIL Image or path to image file
            prompt: Text prompt for LLaVA
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
        """
        # Preprocess the image
        image = self.preprocess_image(image)
        
        try:
            with torch.no_grad():
                # For llava-hf/llava-1.5-7b-hf, image tokens and features need special handling
                if "llava-hf" in self.model_name.lower():
                    # Direct access to model internals to ensure image features are properly passed
                    # This approach bypasses the typical processor workflow to avoid token/feature mismatch
                    vision_tower = getattr(self.model, "vision_tower", None)
                    if vision_tower is not None:
                        # Process the image directly with the vision tower
                        vision_x = self.processor.image_processor(images=image, return_tensors="pt").to(self.device)
                        vision_hidden_states = vision_tower(vision_x.pixel_values).last_hidden_state
                        
                        # Create token inputs without image tokens
                        text_tokens = self.processor.tokenizer([prompt], return_tensors="pt").to(self.device)
                        
                        # Generate response
                        outputs = self.model.generate(
                            **text_tokens,
                            vision_hidden_states=vision_hidden_states,
                            max_new_tokens=max_new_tokens, 
                            do_sample=False
                        )
                        
                        # Decode output
                        generated_text = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    else:
                        # Fallback to standard processor if vision tower not found
                        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
                        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
                    
                # For other processor types
                elif isinstance(self.processor, dict):
                    # For component-based processor
                    pixel_values = self.processor["image_processor"](images=image, return_tensors="pt").pixel_values.to(self.device)
                    input_ids = self.processor["tokenizer"](prompt, return_tensors="pt").input_ids.to(self.device)
                    inputs = {"pixel_values": pixel_values, "input_ids": input_ids}
                    
                    # Generate response
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                    )
                    
                    # Decode output
                    generated_text = self.processor["tokenizer"].decode(output_ids[0], skip_special_tokens=True)
                    
                else:
                    # For standard LlavaProcessor approach
                    inputs = self.processor(
                        text=prompt,
                        images=image,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Generate response
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                    )
                    
                    # Decode output
                    generated_text = self.processor.batch_decode(
                        output_ids, 
                        skip_special_tokens=True
                    )[0]
                
                # Try to extract the response part after the prompt
                if prompt in generated_text:
                    response = generated_text.split(prompt, 1)[1].strip()
                else:
                    # If the prompt is not found, return the full output
                    response = generated_text.strip()
                
                logger.info(f"Generated response: {response[:50]}...")
                return response
                
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise
    
    def infer_from_embedding(self, embedding: torch.Tensor, prompt: str, max_new_tokens: int = 128) -> str:
        """Run LLaVA inference using a pre-computed image embedding.
        
        Args:
            embedding: Pre-computed image embedding tensor
            prompt: Text prompt for LLaVA
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
        """
        try:
            with torch.no_grad():
                # Make sure embedding is a torch tensor
                if not isinstance(embedding, torch.Tensor):
                    raise TypeError(f"Embedding must be a torch.Tensor, got {type(embedding)}")
                
                # Make sure embedding is on the right device
                embedding = embedding.to(self.device)
                
                # Reshape the embedding if needed - LLaVA expects specific dimensions
                if len(embedding.shape) == 1:
                    # Single vector - reshape to [1, dim]
                    embedding = embedding.unsqueeze(0)
                
                # Create token inputs
                logger.info(f"Tokenizing prompt: {prompt[:30]}...")
                text_tokens = self.processor.tokenizer([prompt], return_tensors="pt").to(self.device)
                
                # For LLaVA models, we need to understand the shape needed
                # Try to inspect the vision tower to understand the expected shape
                vision_tower = getattr(self.model, "vision_tower", None)
                expected_embedding_dim = None
                
                # Try to determine expected shape from model configuration
                if hasattr(self.model.config, "vision_config"):
                    if hasattr(self.model.config.vision_config, "hidden_size"):
                        expected_embedding_dim = self.model.config.vision_config.hidden_size
                
                # If we found the expected dimension, reshape if needed
                if expected_embedding_dim is not None:
                    # If last dimension doesn't match expected dim, we need to project
                    if embedding.shape[-1] != expected_embedding_dim:
                        logger.warning(f"Embedding dimension {embedding.shape[-1]} doesn't match expected {expected_embedding_dim}. Attempting simple reshape.")
                        
                        # Try simple rescaling to match dimensions (can be inaccurate but faster than training a projection)
                        # Resize by either repeating or averaging to match expected dimension
                        if embedding.shape[-1] < expected_embedding_dim:
                            # Repeat values to expand
                            repeat_factor = expected_embedding_dim // embedding.shape[-1]
                            remainder = expected_embedding_dim % embedding.shape[-1]
                            
                            if repeat_factor > 0:
                                embedding = embedding.repeat_interleave(repeat_factor, dim=-1)
                                if remainder > 0:
                                    # Add the remaining elements
                                    embedding = torch.cat([embedding, embedding[..., :remainder]], dim=-1)
                        else:
                            # Average values to reduce
                            # Reshape to (batch, seq_len, num_chunks, chunk_size) and average over chunks
                            reshape_size = embedding.shape[-1] // expected_embedding_dim
                            if reshape_size > 0:
                                new_shape = list(embedding.shape[:-1]) + [reshape_size, expected_embedding_dim]
                                embedding = embedding.view(*new_shape).mean(dim=-2)
                            else:
                                # If we can't use exact division, use adaptive pooling
                                embedding = torch.nn.functional.adaptive_avg_pool1d(
                                    embedding.unsqueeze(0), expected_embedding_dim
                                ).squeeze(0)
                
                logger.info(f"Using embedding with shape: {embedding.shape}")
                
                # We'll try several approaches to integrate embeddings with the LLaVA model
                try:
                    # Most direct approach - use as image embeddings with processor for decoding
                    # This is simplest and most CPU-friendly
                    logger.info("Generating with direct embedding approach")
                    
                    # For simple approach, just try using the tokenizer for both input and output
                    outputs = self.model.generate(
                        **text_tokens,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        image_embeds=embedding.unsqueeze(0) if len(embedding.shape) == 1 else embedding
                    )
                    
                    generated_text = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                except (TypeError, ValueError, RuntimeError, AttributeError) as e1:
                    logger.warning(f"First approach failed: {e1}, trying alternative approach")
                    
                    # Second approach - if the model has a direct image embedding input
                    try:
                        logger.info("Generating with image features approach")
                        outputs = self.model.generate(
                            **text_tokens,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            image_features=embedding.unsqueeze(0) if len(embedding.shape) == 1 else embedding
                        )
                        generated_text = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    except (TypeError, ValueError, RuntimeError, AttributeError) as e2:
                        logger.warning(f"Second approach failed: {e2}, using pixel_values approach")
                        
                        # Last approach - treat embedding as pixel values
                        # This is a fallback that's less accurate but might work
                        outputs = self.model.generate(
                            **text_tokens,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            pixel_values=embedding.unsqueeze(0) if len(embedding.shape) == 1 else embedding
                        )
                        generated_text = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract the response part after the prompt
                if prompt in generated_text:
                    response = generated_text.split(prompt, 1)[1].strip()
                    logger.info(f"Found prompt delimiter in response, extracting response part")
                else:
                    # Return the full output if prompt not found
                    response = generated_text.strip()
                    logger.info(f"No prompt delimiter found, returning full response")
                
                logger.info(f"Generated response length: {len(response)} chars")
                return response
                
        except Exception as e:
            logger.error(f"Error during inference from embedding: {str(e)}")
            raise
    
    def batch_process(self, 
                      metadata: List[Dict[str, Any]], 
                      prompts: Dict[str, str] = None,
                      image_key: str = "image_path",
                      show_progress: bool = True) -> List[Dict[str, Any]]:
        """Process a batch of images and add LLaVA outputs to metadata.
        
        Args:
            metadata: List of metadata dicts (each with image_path)
            prompts: Dict mapping output keys to prompt templates
                     (defaults to standard prompts)
            image_key: Key in metadata dict containing image path
            show_progress: Whether to show a progress bar
            
        Returns:
            Updated metadata with LLaVA outputs
        """
        if prompts is None:
            prompts = {
                "llava_caption": PROMPT_TEMPLATES["caption"],
                "llava_scene": PROMPT_TEMPLATES["scene"],
                "llava_table": PROMPT_TEMPLATES["table"],
                "llava_qa": PROMPT_TEMPLATES["qa"],
                "llava_explanation": PROMPT_TEMPLATES["explanation"]
            }
            
        iterator = tqdm(metadata) if show_progress else metadata
        
        for entry in iterator:
            if image_key not in entry:
                continue
                
            image_path = entry[image_key]
            if not os.path.exists(image_path):
                continue
                
            for key, prompt in prompts.items():
                try:
                    entry[key] = self.infer(image_path, prompt)
                except Exception as e:
                    error_msg = f"Error processing image {image_path} with prompt '{key}': {str(e)}"
                    logger.error(error_msg)
                    entry[key] = f"Error: {str(e)}"
                
        return metadata

    def batch_process_embeddings(self,
                                metadata: List[Dict[str, Any]],
                                prompts: Dict[str, str] = None,
                                embedding_key: str = "embedding",
                                show_progress: bool = True) -> List[Dict[str, Any]]:
        """Process a batch of pre-computed embeddings and add LLaVA outputs to metadata.
        
        Args:
            metadata: List of metadata dicts (each with embedding)
            prompts: Dict mapping output keys to prompt templates
                     (defaults to standard prompts)
            embedding_key: Key in metadata dict containing embedding tensor
            show_progress: Whether to show a progress bar
            
        Returns:
            Updated metadata with LLaVA outputs
        """
        if prompts is None:
            prompts = {
                "llava_caption": PROMPT_TEMPLATES["caption"],
                "llava_scene": PROMPT_TEMPLATES["scene"],
                "llava_qa": PROMPT_TEMPLATES["qa"],
                "llava_explanation": PROMPT_TEMPLATES["explanation"]
            }
            
        iterator = tqdm(metadata) if show_progress else metadata
        
        for entry in iterator:
            if embedding_key not in entry:
                logger.warning(f"Embedding key '{embedding_key}' not found in entry")
                continue
                
            embedding = entry[embedding_key]
            
            for key, prompt in prompts.items():
                try:
                    entry[key] = self.infer_from_embedding(embedding, prompt)
                except Exception as e:
                    error_msg = f"Error processing embedding with prompt '{key}': {str(e)}"
                    logger.error(error_msg)
                    entry[key] = f"Error: {str(e)}"
                
        return metadata

# Simple function-based API for those who prefer functions over classes
def load_llava_model(model_name: str = "llava-hf/llava-1.5-7b-hf", device: Optional[str] = None):
    """Load LLaVA model and processor."""
    processor = LLaVAProcessor(model_name=model_name, device=device)
    return processor 