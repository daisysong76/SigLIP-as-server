# CLIP Service Configuration

This directory contains the configuration system for the CLIP service.

## Configuration Files

The service supports both YAML and JSON configuration files:

- `example.yaml`: Example configuration in YAML format
- `example.json`: Equivalent configuration in JSON format

## Configuration System Components

- `loader.py`: Loads configuration from YAML or JSON files
- `validator.py`: Validates configuration against the expected schema
- `schema.py`: Defines the configuration schema using Pydantic models

## Usage

```python
from config.loader import load_config

# Load from a specific file
config = load_config("/path/to/config.yaml")

# Load from environment variable CLIP_CONFIG_PATH or default
config = load_config()

# Access configuration values
server_host = config.server.host
model_batch_size = config.models[0].batch_size
```

## Configuration Structure

### Server Settings

```yaml
server:
  host: 0.0.0.0
  port: 8080
  debug: false
  workers: 8
  timeout: 120
```

### Model Configurations

```yaml
models:
  - name: openai/clip-vit-base-patch32
    device: cuda:0
    batch_size: 32
    max_batch_size: 64
    cache_embeddings: true
    cache_dir: /tmp/clip-cache
    parameters:
      pretrained: openai
      image_size: 224
      normalize: true
```

### Logging Configuration

```yaml
logging:
  level: INFO
  file: /var/log/clip-service.log
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Environment Variables

- `CLIP_CONFIG_PATH`: Path to the configuration file (defaults to internal defaults) 