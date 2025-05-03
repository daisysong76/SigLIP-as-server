"""
Configuration validator for CLIP service.

This module provides validation functionality for CLIP service configurations
using Pydantic models.
"""

from enum import Enum
from typing import Dict, List, Optional, Union, Any

from pydantic import BaseModel, Field, validator, root_validator

class LogLevel(str, Enum):
    """Valid log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class ServerConfig(BaseModel):
    """Server configuration schema."""
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    debug: bool = Field(default=False, description="Enable debug mode")
    workers: int = Field(default=4, ge=1, description="Number of worker processes")
    timeout: int = Field(default=60, ge=1, description="Request timeout in seconds")

class ModelParameters(BaseModel):
    """Model-specific parameters."""
    pretrained: str = Field(default="openai", description="Pretrained model version")
    initial_batch_size: int = Field(default=1, ge=1, description="Initial batch size")
    image_size: int = Field(default=224, ge=32, description="Image size for processing")
    normalize: bool = Field(default=True, description="Whether to normalize embeddings")

class ModelConfig(BaseModel):
    """Model configuration schema."""
    name: str = Field(..., description="Model name or path")
    device: str = Field(default="cuda", description="Device to run model on")
    batch_size: int = Field(default=32, ge=1, description="Default batch size")
    max_batch_size: int = Field(default=64, ge=1, description="Maximum batch size")
    cache_embeddings: bool = Field(default=False, description="Whether to cache embeddings")
    cache_dir: Optional[str] = Field(default=None, description="Cache directory")
    parameters: ModelParameters = Field(default_factory=ModelParameters, description="Model parameters")

    @validator('device')
    def validate_device(cls, v):
        """Validate device string."""
        valid_devices = ['cpu', 'cuda'] + [f'cuda:{i}' for i in range(8)]
        if v not in valid_devices and not v.startswith('cuda:'):
            raise ValueError(f"Device must be one of {valid_devices} or a valid CUDA device")
        return v
    
    @root_validator
    def validate_cache(cls, values):
        """Validate cache configuration."""
        cache_embeddings = values.get('cache_embeddings', False)
        cache_dir = values.get('cache_dir')
        
        if cache_embeddings and not cache_dir:
            raise ValueError("cache_dir must be provided when cache_embeddings is True")
            
        return values

class LoggingConfig(BaseModel):
    """Logging configuration schema."""
    level: LogLevel = Field(default=LogLevel.INFO, description="Log level")
    file: Optional[str] = Field(default=None, description="Log file path")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )

class Config(BaseModel):
    """Main configuration schema."""
    server: ServerConfig = Field(default_factory=ServerConfig, description="Server configuration")
    models: List[ModelConfig] = Field(..., min_items=1, description="Model configurations")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")

    @validator('models')
    def validate_models(cls, v):
        """Validate that model names are unique."""
        model_names = [model.name for model in v]
        if len(model_names) != len(set(model_names)):
            raise ValueError("Model names must be unique")
        return v

def validate_config(config: Dict[str, Any]) -> Config:
    """
    Validate configuration against the schema.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated Config object
        
    Raises:
        ValidationError: If validation fails
    """
    return Config(**config) 