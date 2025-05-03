#!/usr/bin/env python3
"""
Test script for the CLIP service configuration system.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from config import load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_config")

def display_config(config):
    """Display a configuration object in a readable format."""
    logger.info("=== Configuration ===")
    
    # Server config
    logger.info("SERVER:")
    for key, value in config.server.dict().items():
        logger.info(f"  {key}: {value}")
    
    # Models config
    logger.info("MODELS:")
    for i, model in enumerate(config.models):
        logger.info(f"  Model #{i+1}: {model.name}")
        logger.info(f"    device: {model.device}")
        logger.info(f"    batch_size: {model.batch_size}")
        if model.max_batch_size:
            logger.info(f"    max_batch_size: {model.max_batch_size}")
        logger.info(f"    cache_embeddings: {model.cache_embeddings}")
        if model.cache_dir:
            logger.info(f"    cache_dir: {model.cache_dir}")
        
        # Model parameters
        logger.info("    parameters:")
        for key, value in model.parameters.dict().items():
            if value is not None:
                logger.info(f"      {key}: {value}")
    
    # Logging config
    logger.info("LOGGING:")
    for key, value in config.logging.dict().items():
        logger.info(f"  {key}: {value}")

def main():
    parser = argparse.ArgumentParser(description="Test CLIP service configuration")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--json", action="store_true", help="Output configuration as JSON")
    args = parser.parse_args()
    
    try:
        # Load config from file or default
        config = load_config(args.config)
        
        # Display the config
        if args.json:
            print(json.dumps(config.dict(), indent=2))
        else:
            display_config(config)
            
        logger.info("Configuration test successful!")
        return 0
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 