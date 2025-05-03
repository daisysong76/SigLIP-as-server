#!/usr/bin/env python3
"""
CLIP Service Entry Point

This is the main entry point for the CLIP service, connecting all components
and handling runtime flags.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import Config, Server

from api.app import create_app
from inference.pipeline import InferencePipeline
from models.registry import ModelRegistry
from monitoring.metrics import setup_metrics
from optimization.compile import setup_compile_options
from optimization.quantization import setup_quantization


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CLIP Inference Service")
    
    # Server configuration
    parser.add_argument("--host", type=str, help="Host to bind the server to")
    parser.add_argument("--port", type=int, help="Port to bind the server to")
    parser.add_argument("--workers", type=int, help="Number of worker processes")
    
    # Model configuration
    parser.add_argument("--model", type=str, help="CLIP model to use")
    parser.add_argument("--device", type=str, help="Device to run model on (cuda, cpu)")
    parser.add_argument("--precision", type=str, help="Precision to use (fp32, fp16, int8)")
    
    # Optimization flags
    parser.add_argument("--compile", action="store_true", help="Enable TorchCompile")
    parser.add_argument("--no-compile", action="store_false", dest="compile", help="Disable TorchCompile")
    parser.add_argument("--quantize", action="store_true", help="Enable quantization")
    parser.add_argument("--no-quantize", action="store_false", dest="quantize", help="Disable quantization")
    parser.add_argument("--batch-size", type=int, help="Maximum batch size")
    
    # Config file
    parser.add_argument("--config", type=str, default="config/config.yaml", 
                        help="Path to configuration file")
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="server", 
                        choices=["server", "worker", "benchmark"],
                        help="Service mode")
    
    # Debugging
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    
    return parser.parse_args()


def load_config(config_path, args):
    """Load and merge configuration from file and command line arguments."""
    # Load config from file
    config_path = Path(config_path)
    if not config_path.exists():
        logging.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Override with command line arguments if provided
    if args.host:
        config["service"]["host"] = args.host
    if args.port:
        config["service"]["port"] = args.port
    if args.workers:
        config["service"]["workers"] = args.workers
    if args.model:
        config["model"]["name"] = args.model
    if args.device:
        config["model"]["device"] = args.device
    if args.precision:
        config["model"]["precision"] = args.precision
    if args.compile is not None:
        config["model"]["compile"] = args.compile
    if args.quantize is not None:
        config["optimization"]["enable_quantization"] = args.quantize
    if args.batch_size:
        config["inference"]["batch_size"] = args.batch_size
    
    if args.debug:
        config["service"]["log_level"] = "DEBUG"
    
    return config


def setup_logging(config):
    """Set up logging configuration."""
    log_level = getattr(logging, config["service"]["log_level"])
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger("clip-service")


def setup_model(config):
    """Set up and initialize the model."""
    logger = logging.getLogger("clip-service.model")
    logger.info(f"Initializing model: {config['model']['name']}")
    
    # Initialize model registry
    registry = ModelRegistry(config)
    
    # Apply optimization if enabled
    if config["model"]["compile"]:
        setup_compile_options(config)
        
    if config["optimization"]["enable_quantization"]:
        setup_quantization(config)
    
    # Load model
    model = registry.get_model(config["model"]["name"])
    
    return model


def run_server(config, model):
    """Run the FastAPI server."""
    logger = logging.getLogger("clip-service.server")
    logger.info("Starting server")
    
    # Create inference pipeline
    inference_pipeline = InferencePipeline(model, config)
    
    # Setup metrics if enabled
    if config["monitoring"]["metrics"]["enabled"]:
        setup_metrics()
    
    # Create FastAPI app
    app = create_app(config, inference_pipeline)
    
    # CORS setup
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config["api"]["cors_origins"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Run server
    server_config = Config(
        app=app,
        host=config["service"]["host"],
        port=config["service"]["port"],
        workers=config["service"]["workers"],
        log_level=config["service"]["log_level"].lower(),
    )
    server = Server(server_config)
    
    try:
        logger.info(f"Server running at http://{config['service']['host']}:{config['service']['port']}")
        server.run()
    except KeyboardInterrupt:
        logger.info("Server shutting down")


def run_benchmark(config, model):
    """Run benchmarking mode."""
    from optimization.profiler import run_profiling
    
    logger = logging.getLogger("clip-service.benchmark")
    logger.info("Starting benchmark")
    
    # Create inference pipeline
    inference_pipeline = InferencePipeline(model, config)
    
    # Run profiling
    run_profiling(inference_pipeline, config)


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config, args)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting CLIP Service")
    
    # Load model
    model = setup_model(config)
    
    # Run in selected mode
    if args.mode == "server":
        run_server(config, model)
    elif args.mode == "benchmark":
        run_benchmark(config, model)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()


"""
    🚀 How to Run It
Make sure your virtual environment is activated, then just run:
python main.py

You should see:
INFO:     Started server process...
INFO:     Uvicorn running on http://0.0.0.0:8000

Test the health check:
curl http://localhost:8000/health/
# → {"status":"ok"}
    """