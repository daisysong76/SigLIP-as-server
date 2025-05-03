#!/usr/bin/env python3
"""
Configuration schema for the CLIP service using Pydantic for validation.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, validator


class ServerConfig(BaseModel):
    """Server configuration schema."""
    host: str = Field(default="0.0.0.0", description="Server host address")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    debug: bool = Field(default=False, description="Enable debug mode")
    workers: int = Field(default=4, ge=1, le=32, description="Number of worker processes")
    timeout: int = Field(default=60, ge=1, description="Request timeout in seconds")


class ModelParameters(BaseModel):
    """Model-specific parameters schema."""
    pretrained: str = Field(default="webli", description="Pretrained model weights identifier")
    initial_batch_size: int = Field(default=2, ge=1, description="Initial batch size for model loading")
    image_size: int = Field(default=224, ge=32, description="Input image size")
    normalize: bool = Field(default=True, description="Apply normalization to embeddings")


class ModelConfig(BaseModel):
    """Model configuration schema."""
    name: str = Field(description="Model architecture name")
    device: Literal["cpu", "cuda", "mps"] = Field(default="cuda", description="Device for model execution")
    batch_size: int = Field(default=32, ge=1, description="Default inference batch size")
    max_batch_size: int = Field(default=64, ge=1, description="Maximum inference batch size")
    cache_embeddings: bool = Field(default=True, description="Enable embedding caching")
    cache_dir: Optional[str] = Field(default="/tmp/clip-service/cache", description="Directory for cached embeddings")
    parameters: ModelParameters = Field(default_factory=ModelParameters, description="Model-specific parameters")
    
    @validator("max_batch_size")
    def max_batch_size_must_be_greater_than_batch_size(cls, v, values):
        """Validate that max_batch_size is greater than or equal to batch_size."""
        if "batch_size" in values and v < values["batch_size"]:
            raise ValueError("max_batch_size must be greater than or equal to batch_size")
        return v


class LoggingConfig(BaseModel):
    """Logging configuration schema."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    file: Optional[str] = Field(default=None, description="Log file path")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )


class Config(BaseModel):
    """Main configuration schema for the CLIP service."""
    server: ServerConfig = Field(default_factory=ServerConfig, description="Server configuration")
    model: ModelConfig = Field(..., description="Model configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")