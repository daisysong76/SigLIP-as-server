import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, validator, Field

logger = logging.getLogger(__name__)

class ModelParameters(BaseModel):
    pretrained: Optional[str] = "openai"
    image_size: int = 224
    initial_batch_size: Optional[int] = None
    normalize: bool = True

class ModelConfig(BaseModel):
    name: str
    device: str
    batch_size: int = 32
    max_batch_size: Optional[int] = None
    cache_embeddings: bool = False
    cache_dir: Optional[str] = None
    parameters: ModelParameters

    @validator("max_batch_size")
    def validate_max_batch_size(cls, v, values):
        if v is not None and v < values.get("batch_size", 0):
            raise ValueError("max_batch_size must be greater than or equal to batch_size")
        return v

    @validator("cache_dir")
    def validate_cache_dir(cls, v, values):
        if values.get("cache_embeddings", False) and not v:
            logger.warning("cache_embeddings is enabled but no cache_dir specified")
        return v

class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    workers: int = 4
    timeout: int = 60

    @validator("workers")
    def validate_workers(cls, v):
        if v < 1:
            raise ValueError("workers must be greater than 0")
        return v
    
    @validator("timeout")
    def validate_timeout(cls, v):
        if v < 0:
            raise ValueError("timeout cannot be negative")
        return v

class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: Optional[str] = None
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @validator("level")
    def validate_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid logging level. Must be one of {valid_levels}")
        return v.upper()

class Config(BaseModel):
    server: ServerConfig
    models: List[ModelConfig]
    logging: LoggingConfig

    @validator("models")
    def validate_models(cls, v):
        if not v:
            raise ValueError("At least one model configuration must be provided")
        return v

def validate_config(config_dict: Dict[str, Any]) -> Config:
    """
    Validate configuration dictionary against the defined schema.
    
    Args:
        config_dict: Dictionary containing configuration values
        
    Returns:
        Config: Validated configuration object
        
    Raises:
        ValidationError: If configuration doesn't match the schema
    """
    return Config.parse_obj(config_dict) 