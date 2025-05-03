#!/usr/bin/env python3
"""
Configuration loader for CLIP service.
"""

import os
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union

from config.validate import validate_config, Config


logger = logging.getLogger(__name__)


def load_yaml_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        file_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration values
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        yaml.YAMLError: If there's an error parsing the YAML file
    """
    file_path = Path(file_path)
    logger.info(f"Loading YAML configuration from {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {file_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise


def load_json_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        file_path: Path to the JSON configuration file
        
    Returns:
        Dictionary containing configuration values
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If there's an error parsing the JSON file
    """
    file_path = Path(file_path)
    logger.info(f"Loading JSON configuration from {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON configuration: {e}")
        raise


def load_config_from_file(file_path: Union[str, Path]) -> Config:
    """
    Load configuration from a file, detecting format by extension.
    
    Args:
        file_path: Path to the configuration file (YAML or JSON)
        
    Returns:
        Validated configuration object
        
    Raises:
        ValueError: If the file extension is not supported
        ValidationError: If the configuration is invalid
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    if extension in ['.yaml', '.yml']:
        config_dict = load_yaml_config(file_path)
    elif extension == '.json':
        config_dict = load_json_config(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {extension}. Expected .yaml, .yml, or .json")
    
    return validate_config(config_dict)


def load_default_config() -> Config:
    """
    Returns a default configuration.
    
    Returns:
        Validated default configuration
    """
    default_config = {
        "server": {
            "host": "127.0.0.1",
            "port": 8000,
            "debug": False,
            "workers": 4,
            "timeout": 60
        },
        "models": [
            {
                "name": "openai/clip-vit-base-patch32",
                "device": "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
                "batch_size": 32,
                "parameters": {
                    "image_size": 224,
                    "normalize": True
                }
            }
        ],
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }
    
    logger.info("Loading default configuration")
    return validate_config(default_config)


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration from a file or environment variable.
    
    Args:
        config_path: Path to the configuration file (optional)
        
    Returns:
        Validated configuration object
        
    Notes:
        If config_path is None, will look for configuration path in 
        CLIP_SERVICE_CONFIG environment variable. If that's not set,
        will return default configuration.
    """
    if config_path is None:
        config_path = os.environ.get("CLIP_SERVICE_CONFIG")
    
    if config_path:
        try:
            return load_config_from_file(config_path)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.warning("Falling back to default configuration")
            return load_default_config()
    else:
        logger.info("No configuration file specified, using defaults")
        return load_default_config() 