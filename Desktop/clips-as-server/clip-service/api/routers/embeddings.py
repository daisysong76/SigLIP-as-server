"""
Embeddings Router

This module defines the FastAPI router for embedding generation endpoints.
"""
import logging
from typing import Dict, List, Optional, Union

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from starlette.responses import Response

logger = logging.getLogger(__name__)

router = APIRouter()


class TextEmbeddingRequest(BaseModel):
    """Request model for text embedding generation."""
    text: Union[str, List[str]] = Field(..., description="Text to embed")
    model: Optional[str] = Field(None, description="CLIP model to use")
    normalize: bool = Field(True, description="Whether to normalize embeddings")


class ImageEmbeddingRequest(BaseModel):
    """Request model for image embedding generation."""
    # For JSON API when image URLs are provided
    image_urls: Optional[List[str]] = Field(None, description="Image URLs to embed")
    model: Optional[str] = Field(None, description="CLIP model to use")
    normalize: bool = Field(True, description="Whether to normalize embeddings")


class MultimodalEmbeddingRequest(BaseModel):
    """Request model for multimodal embedding generation."""
    text: Optional[Union[str, List[str]]] = Field(None, description="Text to embed")
    image_urls: Optional[List[str]] = Field(None, description="Image URLs to embed")
    model: Optional[str] = Field(None, description="CLIP model to use")
    normalize: bool = Field(True, description="Whether to normalize embeddings")


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""
    embeddings: List[List[float]] = Field(..., description="Generated embeddings")
    model: str = Field(..., description="CLIP model used")
    dimensions: int = Field(..., description="Embedding dimensions")


async def get_inference_pipeline(request: Request):
    """Dependency to get the inference pipeline."""
    return request.app.state.inference_pipeline


@router.post("/text", response_model=EmbeddingResponse)
async def create_text_embeddings(
    request: TextEmbeddingRequest,
    inference_pipeline=Depends(get_inference_pipeline),
):
    """Generate embeddings for text."""
    try:
        logger.info(f"Generating embeddings for text: {request.text}")
        
        # Handle both single text and batch
        texts = request.text if isinstance(request.text, list) else [request.text]
        
        # Generate embeddings
        result = await inference_pipeline.encode_text(
            texts, 
            model_name=request.model, 
            normalize=request.normalize
        )
        
        # Convert to float for JSON serialization
        embeddings = result["embeddings"].tolist()
        
        # Return response
        return {
            "embeddings": embeddings,
            "model": result["model"],
            "dimensions": result["dimensions"],
        }
    except Exception as e:
        logger.error(f"Error generating text embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/image", response_model=EmbeddingResponse)
async def create_image_embeddings(
    request: Optional[ImageEmbeddingRequest] = None,
    files: Optional[List[UploadFile]] = File(None),
    model: Optional[str] = Form(None),
    normalize: bool = Form(True),
    inference_pipeline=Depends(get_inference_pipeline),
):
    """Generate embeddings for images."""
    try:
        # Handle both URL-based and file upload approaches
        if request and request.image_urls:
            logger.info(f"Generating embeddings for image URLs: {request.image_urls}")
            model_name = request.model
            normalize_emb = request.normalize
            
            # Generate embeddings from URLs
            result = await inference_pipeline.encode_image_from_urls(
                request.image_urls, 
                model_name=model_name, 
                normalize=normalize_emb
            )
        elif files:
            logger.info(f"Generating embeddings for {len(files)} uploaded images")
            
            # Read image data
            images = []
            for file in files:
                content = await file.read()
                images.append(content)
            
            # Generate embeddings from binary data
            result = await inference_pipeline.encode_image_from_bytes(
                images, 
                model_name=model, 
                normalize=normalize
            )
        else:
            raise HTTPException(status_code=400, detail="No images provided")
        
        # Convert to float for JSON serialization
        embeddings = result["embeddings"].tolist()
        
        # Return response
        return {
            "embeddings": embeddings,
            "model": result["model"],
            "dimensions": result["dimensions"],
        }
    except Exception as e:
        logger.error(f"Error generating image embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multimodal", response_model=EmbeddingResponse)
async def create_multimodal_embeddings(
    request: MultimodalEmbeddingRequest,
    inference_pipeline=Depends(get_inference_pipeline),
):
    """Generate combined embeddings for text and images."""
    try:
        logger.info(f"Generating multimodal embeddings")
        
        if not request.text and not request.image_urls:
            raise HTTPException(status_code=400, detail="No text or images provided")
        
        # Handle text
        text_embeddings = None
        if request.text:
            texts = request.text if isinstance(request.text, list) else [request.text]
            text_result = await inference_pipeline.encode_text(
                texts, 
                model_name=request.model, 
                normalize=request.normalize
            )
            text_embeddings = text_result["embeddings"]
        
        # Handle images
        image_embeddings = None
        if request.image_urls:
            image_result = await inference_pipeline.encode_image_from_urls(
                request.image_urls, 
                model_name=request.model, 
                normalize=request.normalize
            )
            image_embeddings = image_result["embeddings"]
        
        # Combine embeddings (average if both modalities are present)
        if text_embeddings is not None and image_embeddings is not None:
            # If multiple texts and images, create all combinations
            combined_embeddings = []
            model_name = text_result["model"]  # Should be the same as image_result["model"]
            dimensions = text_result["dimensions"]
            
            # For each text embedding, combine with each image embedding
            for text_emb in text_embeddings:
                for img_emb in image_embeddings:
                    # Simple averaging for combination
                    combined = (text_emb + img_emb) / 2
                    if request.normalize:
                        # Normalize to unit length
                        norm = np.linalg.norm(combined)
                        combined = combined / norm if norm > 0 else combined
                    combined_embeddings.append(combined)
            
            final_embeddings = np.array(combined_embeddings)
        elif text_embeddings is not None:
            final_embeddings = text_embeddings
            model_name = text_result["model"]
            dimensions = text_result["dimensions"]
        else:
            final_embeddings = image_embeddings
            model_name = image_result["model"]
            dimensions = image_result["dimensions"]
        
        # Return response
        return {
            "embeddings": final_embeddings.tolist(),
            "model": model_name,
            "dimensions": dimensions,
        }
    except Exception as e:
        logger.error(f"Error generating multimodal embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))