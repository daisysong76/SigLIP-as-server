#!/usr/bin/env python3
"""
Model manager for handling multiple model instances.
"""

import os
import torch
import logging
from typing import Dict, Any, Optional, List, Tuple
from ..config.schema import Config, ModelConfig, ModelType
from .loader import ModelLoaderFactory

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages loading and accessing models based on configuration.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the model manager.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.models: Dict[str, Tuple[Any, Any]] = {}  # name -> (model, processor)
        self.default_model_name: Optional[str] = None
        
    def load_models(self):
        """
        Load all models specified in the configuration.
        """
        logger.info(f"Loading {len(self.config.models)} models")
        
        for model_config in self.config.models:
            self._load_model(model_config)
            
            # Set as default if specified
            if model_config.default:
                if self.default_model_name is not None:
                    logger.warning(f"Multiple default models specified. Using {model_config.name} as default.")
                self.default_model_name = model_config.name
        
        # If no default was explicitly set but we have models, use the first one
        if self.default_model_name is None and self.models:
            self.default_model_name = self.config.models[0].name
            logger.info(f"No default model specified. Using {self.default_model_name} as default.")
            
        logger.info(f"Successfully loaded {len(self.models)} models")
    
    def _load_model(self, model_config: ModelConfig):
        """
        Load a single model based on its configuration.
        
        Args:
            model_config: Model configuration
        """
        try:
            logger.info(f"Loading model {model_config.name}")
            
            # Create appropriate loader based on model type
            loader = ModelLoaderFactory.create_loader(model_config.model_type)
            
            # Load model and processor
            model, processor = loader.load(model_config)
            
            # Store model and processor
            self.models[model_config.name] = (model, processor)
            logger.info(f"Model {model_config.name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_config.name}: {str(e)}")
            raise
    
    def get_model(self, name: Optional[str] = None) -> Tuple[Any, Any]:
        """
        Get a model and processor by name.
        
        Args:
            name: Model name (uses default if None)
            
        Returns:
            Tuple of (model, processor)
            
        Raises:
            ValueError: If the model is not found
        """
        # Use default model if name is not specified
        if name is None:
            if self.default_model_name is None:
                raise ValueError("No default model available")
            name = self.default_model_name
        
        # Get model by name
        if name not in self.models:
            raise ValueError(f"Model {name} not found")
        
        return self.models[name]
    
    def get_model_names(self) -> List[str]:
        """
        Get a list of all available model names.
        
        Returns:
            List of model names
        """
        return list(self.models.keys())
    
    def get_default_model_name(self) -> Optional[str]:
        """
        Get the name of the default model.
        
        Returns:
            Default model name or None if no default is set
        """
        return self.default_model_name 