#!/usr/bin/env python3
"""
Model registry for managing multiple CLIP, LLaVA, and SigLIP models.
"""

import os
import logging
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Enum for model types."""
    CLIP = "clip"
    LLAVA = "llava"
    SIGLIP = "siglip"

@dataclass
class ModelConfig:
    """Configuration for a model in the registry."""
    name: str
    model_path: str
    model_type: ModelType
    device: str = "cuda"
    load_8bit: bool = False
    load_4bit: bool = False
    initial_batch_size: int = 4
    max_batch_size: int = 32
    cache_embeddings: bool = True
    description: str = ""

class ModelRegistry:
    """
    Registry for managing multiple CLIP, LLaVA and SigLIP models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model registry.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models = {}
        self.default_model_name = None
        
        # Initialize from configuration
        self._initialize_from_config()
    
    def _initialize_from_config(self):
        """Initialize models from the configuration."""
        model_configs = self.config.get("models", [])
        
        # If no models are specified, create default CLIP model
        if not model_configs:
            logger.info("No models specified in configuration, using default CLIP model")
            model_configs = [{
                "name": "clip-vit-b-32",
                "model_path": "openai/clip-vit-base-patch32",
                "model_type": "clip",
                "device": self.config.get("model", {}).get("device", "cuda"),
                "description": "Default CLIP ViT-B/32 model"
            }]
        
        # Register all models in the configuration
        for model_config in model_configs:
            model_name = model_config.get("name")
            model_path = model_config.get("model_path")
            model_type_str = model_config.get("model_type", "clip").lower()
            
            # Map string model type to enum
            if model_type_str == "clip":
                model_type = ModelType.CLIP
            elif model_type_str == "llava":
                model_type = ModelType.LLAVA
            elif model_type_str == "siglip":
                model_type = ModelType.SIGLIP
            else:
                logger.error(f"Invalid model type: {model_type_str}")
                continue
            
            if not model_name or not model_path:
                logger.error(f"Invalid model configuration: {model_config}")
                continue
            
            # Create model configuration
            config = ModelConfig(
                name=model_name,
                model_path=model_path,
                model_type=model_type,
                device=model_config.get("device", self.config.get("model", {}).get("device", "cuda")),
                load_8bit=model_config.get("load_8bit", self.config.get("optimization", {}).get("enable_8bit", False)),
                load_4bit=model_config.get("load_4bit", self.config.get("optimization", {}).get("enable_4bit", False)),
                initial_batch_size=model_config.get("batch_size", self.config.get("inference", {}).get("batch_size", 4)),
                max_batch_size=model_config.get("max_batch_size", self.config.get("inference", {}).get("max_batch_size", 32)),
                cache_embeddings=model_config.get("cache_embeddings", self.config.get("inference", {}).get("cache_embeddings", True)),
                description=model_config.get("description", "")
            )
            
            # Register model configuration
            self.register_model_config(config)
            
            # Set default model if not set
            if self.default_model_name is None:
                self.default_model_name = model_name
            # Or if explicitly marked as default
            if model_config.get("default", False):
                self.default_model_name = model_name
    
    def register_model_config(self, config: ModelConfig):
        """
        Register a model configuration.
        
        Args:
            config: Model configuration
        """
        if config.name in self.models:
            logger.warning(f"Model {config.name} already registered, will be overwritten")
        
        self.models[config.name] = {
            "config": config,
            "instance": None
        }
        logger.info(f"Registered model configuration for {config.name} ({config.model_type.value}) using {config.model_path}")
    
    def get_model(self, model_name: Optional[str] = None) -> Any:
        """
        Get a model by name. If the model is not loaded, it will be loaded.
        
        Args:
            model_name: Name of the model to get. If None, the default model will be used.
            
        Returns:
            The model instance
        """
        # Use default model if not specified
        model_name = model_name or self.default_model_name
        
        # Check if model is registered
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not registered")
        
        # Load model if not loaded
        if self.models[model_name]["instance"] is None:
            self._load_model(model_name)
        
        return self.models[model_name]["instance"]
    
    def _load_model(self, model_name: str):
        """
        Load a model by name.
        
        Args:
            model_name: Name of the model to load
        """
        model_data = self.models[model_name]
        config = model_data["config"]
        
        logger.info(f"Loading model {model_name} ({config.model_type.value}) from {config.model_path}")
        
        # Load model based on type
        if config.model_type == ModelType.CLIP:
            self._load_clip_model(model_name)
        elif config.model_type == ModelType.LLAVA:
            self._load_llava_model(model_name)
        elif config.model_type == ModelType.SIGLIP:
            self._load_siglip_model(model_name)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
    
    def _load_clip_model(self, model_name: str):
        """
        Load a CLIP model.
        
        Args:
            model_name: Name of the model to load
        """
        try:
            import clip
            import torch
            from model.clip_model import CLIPModelWrapper
            
            model_data = self.models[model_name]
            config = model_data["config"]
            
            # Load CLIP model
            model_instance = CLIPModelWrapper(
                model_name=config.model_path,
                device=config.device,
                initial_batch_size=config.initial_batch_size,
                max_batch_size=config.max_batch_size,
                cache_embeddings=config.cache_embeddings
            )
            
            # Store model instance
            model_data["instance"] = model_instance
            logger.info(f"Loaded CLIP model {model_name}")
        
        except Exception as e:
            logger.error(f"Error loading CLIP model {model_name}: {e}")
            raise
    
    def _load_llava_model(self, model_name: str):
        """
        Load a LLaVA model.
        
        Args:
            model_name: Name of the model to load
        """
        try:
            from model.llava_model import LLaVAModelWrapper
            
            model_data = self.models[model_name]
            config = model_data["config"]
            
            # Load LLaVA model
            model_instance = LLaVAModelWrapper(
                model_path=config.model_path,
                device=config.device,
                load_8bit=config.load_8bit,
                load_4bit=config.load_4bit,
                initial_batch_size=config.initial_batch_size,
                max_batch_size=config.max_batch_size,
                cache_embeddings=config.cache_embeddings
            )
            
            # Store model instance
            model_data["instance"] = model_instance
            logger.info(f"Loaded LLaVA model {model_name}")
        
        except Exception as e:
            logger.error(f"Error loading LLaVA model {model_name}: {e}")
            raise
    
    def _load_siglip_model(self, model_name: str):
        """
        Load a SigLIP model.
        
        Args:
            model_name: Name of the model to load
        """
        try:
            from model.siglip_model import SigLIPModelWrapper
            
            model_data = self.models[model_name]
            config = model_data["config"]
            
            # Load SigLIP model
            model_instance = SigLIPModelWrapper(
                model_name=config.model_path,
                device=config.device,
                initial_batch_size=config.initial_batch_size,
                max_batch_size=config.max_batch_size,
                cache_embeddings=config.cache_embeddings
            )
            
            # Store model instance
            model_data["instance"] = model_instance
            logger.info(f"Loaded SigLIP model {model_name}")
        
        except Exception as e:
            logger.error(f"Error loading SigLIP model {model_name}: {e}")
            raise
    
    def get_model_names(self) -> List[str]:
        """
        Get a list of registered model names.
        
        Returns:
            List of model names
        """
        return list(self.models.keys())
    
    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            model_name: Name of the model to get info for. If None, gets info for all models.
            
        Returns:
            Dictionary with model information
        """
        if model_name is not None:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not registered")
            
            config = self.models[model_name]["config"]
            is_loaded = self.models[model_name]["instance"] is not None
            
            return {
                "name": config.name,
                "model_path": config.model_path,
                "model_type": config.model_type.value,
                "device": config.device,
                "description": config.description,
                "loaded": is_loaded,
                "is_default": model_name == self.default_model_name
            }
        else:
            return {
                name: {
                    "name": self.models[name]["config"].name,
                    "model_path": self.models[name]["config"].model_path,
                    "model_type": self.models[name]["config"].model_type.value,
                    "device": self.models[name]["config"].device,
                    "description": self.models[name]["config"].description,
                    "loaded": self.models[name]["instance"] is not None,
                    "is_default": name == self.default_model_name
                }
                for name in self.models
            }
    
    def default_model(self) -> Any:
        """
        Get the default model.
        
        Returns:
            The default model instance
        """
        return self.get_model(self.default_model_name)
    
    def unload_model(self, model_name: str):
        """
        Unload a model from memory.
        
        Args:
            model_name: Name of the model to unload
        """
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not registered, cannot unload")
            return
        
        if self.models[model_name]["instance"] is None:
            logger.info(f"Model {model_name} not loaded, no need to unload")
            return
        
        # Delete model instance
        del self.models[model_name]["instance"]
        self.models[model_name]["instance"] = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Unloaded model {model_name}")
    
    def unload_all_models(self):
        """Unload all models from memory."""
        for model_name in self.models:
            self.unload_model(model_name)
        
        logger.info("Unloaded all models") 