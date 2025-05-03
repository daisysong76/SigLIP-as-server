"""Modern dataset configuration for CLIP testing using HuggingFace datasets."""
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, Union, AsyncIterator
from datasets import load_dataset, Dataset, Image, Features, Sequence, concatenate_datasets, IterableDataset
import torch
from transformers import CLIPProcessor
import numpy as np
from pathlib import Path
from itertools import chain
import logging
import asyncio
from functools import partial
from inference.collate_fn import multimodal_collate_fn, image_only_collate_fn, text_only_collate_fn
from PIL import Image

logger = logging.getLogger(__name__)

class AsyncIterableWrapper:
    """Wrapper to make IterableDataset async-compatible."""
    def __init__(self, dataset: IterableDataset):
        self.dataset = dataset
        self._iterator = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._iterator is None:
            self._iterator = iter(self.dataset)
        try:
            # Run iteration in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            item = await loop.run_in_executor(None, next, self._iterator)
            return item
        except StopIteration:
            raise StopAsyncIteration

@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""
    # Image datasets
    image_datasets: List[str] = (
        "nlphuji/flickr30k",  # Well-annotated image dataset with text pairs
    )
    
    # Text datasets
    text_datasets: List[str] = (
        "nlphuji/flickr30k",  # Using same dataset for text to ensure paired data
    )
    
    # Local paths
    local_image_dir: str = "input/images"
    local_text_file: str = "input/text/prompts.txt"
    
    # Processing config
    batch_size: int = 32
    max_text_length: int = 77  # CLIP's max token length
    image_size: int = 224  # CLIP's standard image size
    num_proc: int = 4
    
    async def load_streaming_dataset(
        self, 
        clip_processor: CLIPProcessor, 
        streaming: bool = True
    ) -> Union[Tuple[AsyncIterableWrapper, AsyncIterableWrapper], Tuple[Dataset, Dataset]]:
        """Load streaming dataset with optimized settings."""
        split = "test"  # Using test split since that's what's available
        
        try:
            logger.info(f"Starting to load datasets with streaming={streaming}")
            logger.info(f"Image datasets to load: {self.image_datasets}")
            logger.info(f"Text datasets to load: {self.text_datasets}")
            
            # Load image datasets with advanced error handling
            image_datasets = []
            for name in self.image_datasets:
                try:
                    logger.info(f"Loading image dataset: {name}")
                    ds = load_dataset(name, split=split, streaming=streaming)
                    logger.info(f"Dataset loaded. Column names: {ds.column_names}")
                    
                    if streaming:
                        logger.info("Processing streaming dataset")
                        # FIXED: Process each example with consistent structure (all lists of same length)
                        # When batched=True, all items must have the same length to satisfy HF's validation
                        def process_image(example):
                            """Process a single image example - all outputs must be lists of same length"""
                            # Handle different dataset column naming conventions
                            image_key = "image"
                            if image_key not in example:
                                # Try other common names
                                for key in ["img", "pixel_values", "image_data", "picture"]:
                                    if key in example:
                                        image_key = key
                                        break
                            
                            if image_key not in example:
                                logger.warning(f"Could not find image column in {list(example.keys())}")
                                # Provide a placeholder
                                return {
                                    "image": [None],     # Placeholder
                                    "image_id": [-1]     # Placeholder ID
                                }
                                
                            image_id_hash = hash(str(example[image_key]))
                            return {
                                "image": [example[image_key]],     # Make a length-1 list
                                "image_id": [image_id_hash]        # Make a length-1 list
                            }
                        
                        ds = ds.map(
                            process_image,
                            remove_columns=ds.column_names,
                            batched=False  # Process one example at a time
                        )
                        logger.info("Streaming dataset processed")
                    image_datasets.append(ds)
                    logger.info(f"Successfully loaded and processed {name}")
                except Exception as e:
                    logger.error(f"Failed to load image dataset {name}: {e}", exc_info=True)
                    continue
            
            # Load text datasets similarly
            text_datasets = []
            for name in self.text_datasets:
                try:
                    logger.info(f"Loading text dataset: {name}")
                    ds = load_dataset(name, split=split, streaming=streaming)
                    logger.info(f"Dataset loaded. Column names: {ds.column_names}")
                    
                    if streaming:
                        logger.info("Processing streaming dataset")
                        # FIXED: Process each example with consistent structure (all lists of same length)
                        def process_text(example):
                            """Process a single text example - all outputs must be lists of same length"""
                            # Handle different dataset column naming conventions
                            text_key = None
                            for key in ["text", "caption", "sentence", "description", "label"]:
                                if key in example:
                                    text_key = key
                                    break
                            
                            if text_key is None:
                                # If no text column, try to convert labels to text
                                if "fine_label" in example:
                                    # For CIFAR-100
                                    class_name = str(example["fine_label"])
                                    text_id_hash = hash(class_name)
                                    return {
                                        "text": [f"A {class_name}"],  # Make a length-1 list 
                                        "text_id": [text_id_hash]     # Make a length-1 list
                                    }
                                
                                logger.warning(f"Could not find text column in {list(example.keys())}")
                                # Provide a placeholder
                                return {
                                    "text": ["Unknown"],  # Placeholder
                                    "text_id": [-1]       # Placeholder ID
                                }
                            
                            text_id_hash = hash(str(example[text_key]))
                            return {
                                "text": [example[text_key]],  # Make a length-1 list
                                "text_id": [text_id_hash]     # Make a length-1 list
                            }
                        
                        ds = ds.map(
                            process_text,
                            remove_columns=ds.column_names,
                            batched=False  # Process one example at a time
                        )
                        logger.info("Streaming dataset processed")
                    text_datasets.append(ds)
                    logger.info(f"Successfully loaded and processed {name}")
                except Exception as e:
                    logger.error(f"Failed to load text dataset {name}: {e}", exc_info=True)
                    continue
            
            if not image_datasets or not text_datasets:
                raise ValueError("No datasets were successfully loaded")
            
            logger.info("Combining datasets")
            # Combine datasets efficiently
            if streaming:
                logger.info("Processing streaming datasets for final output")
                
                # FIXED: Ensure consistent batch behavior with map functions
                def preprocess_image_batch(batch):
                    """Preprocess a batch of images with consistent output structure"""
                    try:
                        # Ensure batch contains lists of same length
                        batch_size = len(batch["image"])
                        logger.debug(f"Preprocessing image batch of size {batch_size}")
                        
                        # Convert None images to empty arrays if needed
                        images = []
                        for img in batch["image"]:
                            if img is None:
                                # Create an empty RGB image
                                img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
                            elif isinstance(img, np.ndarray):
                                # Convert numpy array to PIL Image
                                img = Image.fromarray(img)
                            images.append(img)
                        
                        # If images is a list of tensors, stack them early
                        if len(images) > 0:
                            if isinstance(images[0], torch.Tensor):
                                images = torch.stack(images)  # Stack into (batch_size, 3, 224, 224)
                            elif isinstance(images[0], Image.Image):
                                pass  # Fine, let CLIP processor handle it
                            else:
                                logger.warning(f"Unsupported image type: {type(images[0])}")
                        
                        # Process with processor
                        processed = clip_processor(
                            images=images,
                            return_tensors="pt",
                            padding=True
                        )
                        
                        # Ensure output has lists of same length
                        return {
                            "pixel_values": processed.pixel_values.to(torch.float32),
                            "image_id": batch["image_id"]  # Already a list from earlier processing
                        }
                    except Exception as e:
                        logger.error(f"Error in batch preprocessing: {e}", exc_info=True)
                        # Return a simple batch structure with lists of length 1
                        return {
                            "pixel_values": torch.zeros((1, 3, self.image_size, self.image_size), dtype=torch.float32),
                            "image_id": [-1]  # List with one item
                        }
                
                def preprocess_text_batch(batch):
                    """Preprocess a batch of texts with consistent output structure"""
                    try:
                        # Ensure batch contains lists of same length
                        batch_size = len(batch["text"])
                        logger.debug(f"Preprocessing text batch of size {batch_size}")
                        
                        processed = clip_processor(
                            text=batch["text"],
                            return_tensors="pt",
                            padding="max_length",
                            max_length=self.max_text_length,
                            truncation=True
                        )
                        
                        # Ensure output has lists of same length
                        return {
                            "input_ids": processed.input_ids,
                            "attention_mask": processed.attention_mask,
                            "text_id": batch["text_id"]  # Already a list from earlier processing
                        }
                    except Exception as e:
                        logger.error(f"Error in batch preprocessing: {e}", exc_info=True)
                        # Return a simple batch structure with lists of length 1
                        return {
                            "input_ids": torch.zeros((1, self.max_text_length), dtype=torch.long),
                            "attention_mask": torch.zeros((1, self.max_text_length), dtype=torch.long),
                            "text_id": [-1]  # List with one item
                        }
                
                # Apply batch processing
                image_dataset = image_datasets[0].map(
                    preprocess_image_batch,
                    batched=True,
                    batch_size=1  # Process one example at a time for consistent handling
                )
                text_dataset = text_datasets[0].map(
                    preprocess_text_batch,
                    batched=True,
                    batch_size=1  # Process one example at a time for consistent handling
                )
                
                # Wrap datasets in async iterator
                return AsyncIterableWrapper(image_dataset), AsyncIterableWrapper(text_dataset)
            else:
                logger.info("Concatenating non-streaming datasets")
                image_dataset = concatenate_datasets(image_datasets)
                text_dataset = concatenate_datasets(text_datasets)
                return image_dataset, text_dataset
            
        except Exception as e:
            logger.error(f"Error in dataset loading: {e}", exc_info=True)
            raise
    
    def _preprocess_image(self, batch: Dict[str, Any], processor: CLIPProcessor) -> Dict[str, Any]:
        """Preprocess image batch with advanced error handling."""
        try:
            logger.debug(f"Preprocessing image batch of size: {len(batch['image'])}")
            # Process images in batches
            processed = processor(
                images=batch["image"],
                return_tensors="pt",
                padding=True
            )
            logger.debug("Image preprocessing completed")
            
            return {
                "pixel_values": processed.pixel_values.to(torch.float32),
                "image_id": batch["image_id"]
            }
        except Exception as e:
            logger.error(f"Error preprocessing image batch: {e}", exc_info=True)
            # Return empty batch with correct structure
            return {
                "pixel_values": torch.zeros((1, 3, self.image_size, self.image_size), dtype=torch.float32),
                "image_id": [-1]  # List with one item
            }
    
    def _preprocess_text(self, batch: Dict[str, Any], processor: CLIPProcessor) -> Dict[str, Any]:
        """Preprocess text batch with advanced error handling."""
        try:
            logger.debug(f"Preprocessing text batch of size: {len(batch['text'])}")
            # Process text in batches
            processed = processor(
                text=batch["text"],
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_text_length,
                truncation=True
            )
            logger.debug("Text preprocessing completed")
            
            return {
                "input_ids": processed.input_ids,
                "attention_mask": processed.attention_mask,
                "text_id": batch["text_id"]
            }
        except Exception as e:
            logger.error(f"Error preprocessing text batch: {e}", exc_info=True)
            # Return empty batch with correct structure
            return {
                "input_ids": torch.zeros((1, self.max_text_length), dtype=torch.long),
                "attention_mask": torch.zeros((1, self.max_text_length), dtype=torch.long),
                "text_id": [-1]  # List with one item
            }
    
    @staticmethod
    def create_dataloader(
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        pin_memory: bool = True,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        drop_last: bool = False,
        collate_fn: Optional[callable] = None,
        modality: str = "multimodal"
    ) -> torch.utils.data.DataLoader:
        """Create an optimized DataLoader with modern settings."""
        # Select appropriate collate function if not provided
        if collate_fn is None:
            if modality == "image":
                from inference.collate_fn import image_only_collate_fn
                collate_fn = image_only_collate_fn
            elif modality == "text":
                from inference.collate_fn import text_only_collate_fn
                collate_fn = text_only_collate_fn
            else:  # multimodal
                from inference.collate_fn import multimodal_collate_fn
                collate_fn = multimodal_collate_fn
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            drop_last=drop_last,
            collate_fn=collate_fn,
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        ) 