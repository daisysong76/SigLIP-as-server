#!/usr/bin/env python3
"""
Model loader for CLIP, LLaVA and SigLIP models.
"""

import os
import torch
import logging
from typing import Dict, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod

from config.schema import ModelConfig, ModelType

logger = logging.getLogger(__name__)

class ModelLoader(ABC):
    """Abstract base class for model loaders."""
    
    @abstractmethod
    def load(self, config: ModelConfig) -> Tuple[Any, Any]:
        """
        Load a model from a configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            Tuple of (model, processor)
        """
        pass

class CLIPModelLoader(ModelLoader):
    """Loader for CLIP models."""
    
    def load(self, config: ModelConfig) -> Tuple[Any, Any]:
        """
        Load a CLIP model from a configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            Tuple of (model, processor)
        """
        try:
            from transformers import CLIPModel, CLIPProcessor
            
            logger.info(f"Loading CLIP model {config.name} from {config.model_path}")
            
            # Set precision for loading
            torch_dtype = None
            if config.clip_precision == "float16":
                torch_dtype = torch.float16
            elif config.clip_precision == "float32":
                torch_dtype = torch.float32
            
            # Load model and processor
            model = CLIPModel.from_pretrained(
                config.model_path,
                torch_dtype=torch_dtype,
                device_map=config.device if config.device != "cpu" else None
            )
            processor = CLIPProcessor.from_pretrained(config.model_path)
            
            logger.info(f"Successfully loaded CLIP model {config.name}")
            return model, processor
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model {config.name}: {str(e)}")
            raise

class SigLIPModelLoader(ModelLoader):
    """Loader for SigLIP models."""
    
    def load(self, config: ModelConfig) -> Tuple[Any, Any]:
        """
        Load a SigLIP model from a configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            Tuple of (model, processor)
        """
        try:
            from transformers import AutoProcessor, AutoModel
            
            logger.info(f"Loading SigLIP model {config.name} from {config.model_path}")
            
            # Set precision for loading
            torch_dtype = torch.float16 if config.precision == "fp16" else torch.float32
            
            # Load model
            model = AutoModel.from_pretrained(
                config.model_path,
                torch_dtype=torch_dtype,
                device_map=config.device if config.device != "cpu" else None
            )
            
            # Load processor - SigLIP uses similar processor to CLIP
            processor = AutoProcessor.from_pretrained(config.model_path)
            
            logger.info(f"Successfully loaded SigLIP model {config.name}")
            return model, processor
            
        except Exception as e:
            logger.error(f"Failed to load SigLIP model {config.name}: {str(e)}")
            raise

class LLaVAModelLoader(ModelLoader):
    """Loader for LLaVA models."""
    
    def load(self, config: ModelConfig) -> Tuple[Any, Any]:
        """
        Load a LLaVA model from a configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            Tuple of (model, processor)
        """
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
            
            logger.info(f"Loading LLaVA model {config.name} from {config.model_path}")
            
            # Configure quantization if enabled
            quantization_config = None
            if config.load_8bit:
                logger.info(f"Loading {config.name} in 8-bit mode")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
            elif config.load_4bit:
                logger.info(f"Loading {config.name} in 4-bit mode")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            # Load model with appropriate device configuration
            model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                device_map=config.device if config.device != "cpu" else None,
                trust_remote_code=True
            )
            
            # Load processor
            processor = AutoProcessor.from_pretrained(config.model_path, trust_remote_code=True)
            
            logger.info(f"Successfully loaded LLaVA model {config.name}")
            return model, processor
            
        except Exception as e:
            logger.error(f"Failed to load LLaVA model {config.name}: {str(e)}")
            raise

class ModelLoaderFactory:
    """Factory for creating model loaders."""
    
    @staticmethod
    def create_loader(model_type: ModelType) -> ModelLoader:
        """
        Create a model loader for the specified model type.
        
        Args:
            model_type: Type of model to load
            
        Returns:
            Model loader instance
        """
        if model_type == ModelType.CLIP:
            return CLIPModelLoader()
        elif model_type == ModelType.LLAVA:
            return LLaVAModelLoader()
        elif model_type == ModelType.SIGLIP:
            return SigLIPModelLoader()
        else:
            raise ValueError(f"Unsupported model type: {model_type}") 