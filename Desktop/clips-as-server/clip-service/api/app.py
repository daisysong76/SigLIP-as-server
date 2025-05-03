"""
FastAPI Application Module

This module creates and configures the FastAPI application for the CLIP service.
"""
import logging
from typing import Dict, List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.middleware.auth import APIKeyMiddleware
from api.routers import embeddings, health, search

logger = logging.getLogger(__name__)


def create_app(config: Dict, inference_pipeline) -> FastAPI:
    """Create and configure the FastAPI application."""
    # Create app
    app = FastAPI(
        title="CLIP Inference Service",
        description="High-performance CLIP embedding service for multimodal applications",
        version=config["service"]["version"],
        docs_url="/docs" if config["api"]["enable_docs"] else None,
        redoc_url="/redoc" if config["api"]["enable_docs"] else None,
    )
    
    # Add middleware
    if config["api"]["auth"]["enabled"]:
        app.add_middleware(
            APIKeyMiddleware,
            api_key_header=config["api"]["auth"]["api_key_header"],
            api_keys=config["api"]["auth"]["api_keys"],
        )
    
    # Register routers
    app.include_router(
        health.router,
        prefix=f"{config['api']['prefix']}/health",
        tags=["Health"],
    )
    
    app.include_router(
        embeddings.router,
        prefix=f"{config['api']['prefix']}/embeddings",
        tags=["Embeddings"],
    )
    
    app.include_router(
        search.router,
        prefix=f"{config['api']['prefix']}/search",
        tags=["Search"],
    )
    
    # Dependency injection
    app.state.inference_pipeline = inference_pipeline
    app.state.config = config
    
    # Exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "detail": str(exc)},
        )
    
    # Startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting CLIP API Service")
        # Initialize any resources if needed
    
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down CLIP API Service")
        # Clean up resources if needed
    
    return app