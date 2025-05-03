#!/usr/bin/env python3
"""
Test for the configuration loader.
"""

import os
import sys
import tempfile
import json
import yaml

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the configuration loader
from config.loader import (
    load_yaml_config,
    load_json_config,
    load_config_from_file,
    load_default_config,
    load_config
)

def test_yaml_config():
    """Test loading configuration from a YAML file."""
    # Create a temporary YAML file
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w+', delete=False) as temp:
        temp.write("""
server:
  host: '127.0.0.1'
  port: 8080
  debug: true
model:
  name: 'test-model'
  device: 'cpu'
  batch_size: 16
  cache_dir: '/tmp/cache'
        """)
        temp_path = temp.name
    
    try:
        # Load the configuration from the YAML file
        config = load_yaml_config(temp_path)
        
        # Check the configuration values
        assert config['server']['host'] == '127.0.0.1'
        assert config['server']['port'] == 8080
        assert config['server']['debug'] is True
        assert config['model']['name'] == 'test-model'
        assert config['model']['device'] == 'cpu'
        assert config['model']['batch_size'] == 16
        assert config['model']['cache_dir'] == '/tmp/cache'
        
        print("✅ YAML config test passed")
    finally:
        # Clean up the temporary file
        os.unlink(temp_path)

def test_json_config():
    """Test loading configuration from a JSON file."""
    # Create a temporary JSON file
    with tempfile.NamedTemporaryFile(suffix='.json', mode='w+', delete=False) as temp:
        json.dump({
            "server": {
                "host": "0.0.0.0",
                "port": 9000,
                "debug": False
            },
            "model": {
                "name": "json-model",
                "device": "cuda",
                "batch_size": 32,
                "cache_dir": "/tmp/json-cache"
            }
        }, temp)
        temp_path = temp.name
    
    try:
        # Load the configuration from the JSON file
        config = load_json_config(temp_path)
        
        # Check the configuration values
        assert config['server']['host'] == '0.0.0.0'
        assert config['server']['port'] == 9000
        assert config['server']['debug'] is False
        assert config['model']['name'] == 'json-model'
        assert config['model']['device'] == 'cuda'
        assert config['model']['batch_size'] == 32
        assert config['model']['cache_dir'] == '/tmp/json-cache'
        
        print("✅ JSON config test passed")
    finally:
        # Clean up the temporary file
        os.unlink(temp_path)

def test_load_config_from_file():
    """Test loading configuration from different file types."""
    # Create temporary YAML and JSON files
    yaml_path = None
    json_path = None
    
    try:
        # Create a YAML file
        with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w+', delete=False) as temp:
            temp.write("""
server:
  host: 'localhost'
  port: 7000
            """)
            yaml_path = temp.name
        
        # Create a JSON file
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w+', delete=False) as temp:
            json.dump({
                "server": {
                    "host": "jsonhost",
                    "port": 7001
                }
            }, temp)
            json_path = temp.name
        
        # Load from YAML file
        yaml_config = load_config_from_file(yaml_path)
        assert yaml_config.server.host == 'localhost'
        assert yaml_config.server.port == 7000
        
        # Load from JSON file
        json_config = load_config_from_file(json_path)
        assert json_config.server.host == 'jsonhost'
        assert json_config.server.port == 7001
        
        print("✅ load_config_from_file test passed")
    finally:
        # Clean up the temporary files
        if yaml_path:
            os.unlink(yaml_path)
        if json_path:
            os.unlink(json_path)

def test_default_config():
    """Test loading the default configuration."""
    config = load_default_config()
    
    # Check default values
    assert config.server.host == '0.0.0.0'
    assert config.server.port == 8000
    assert config.server.debug is False
    assert config.model.name == 'SigLIP'
    assert config.model.batch_size == 32
    
    print("✅ Default config test passed")

def main():
    """Run all the tests."""
    print("Testing configuration loader...\n")
    
    # Run the tests
    test_yaml_config()
    test_json_config()
    test_load_config_from_file()
    test_default_config()
    
    print("\nAll configuration loader tests passed! ✨")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 