"""Dataset configuration for CLIP model testing."""
from typing import Tuple, Any, Iterable
from dataclasses import dataclass
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import CLIPProcessor
from datasets import interleave_datasets
from datasets import IterableDataset

@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""
    image_datasets: Tuple[str, ...] = (
        'cifar100',     # Public dataset with 100 classes
        'beans',        # Public dataset with plant disease images
        'food101',      # Public dataset with food images
    )
    local_image_dir: str = 'input/images'
    local_text_file: str = 'input/text/prompts.txt'
    batch_size: int = 32
    max_text_length: int = 77  # CLIP's max context length
    image_size: int = 224      # ViT-B/32 input size
    num_proc: int = 4          # Number of processes for data loading

    def load_streaming_dataset(
        self,
        processor: CLIPProcessor,
        streaming: bool = True,
        split: str = 'train[:100]'  # Limit to 100 examples for testing
    ) -> Dataset:
        """Load and preprocess streaming dataset."""
        # Load only CIFAR100 for testing
        dataset = load_dataset('cifar100', split=split, streaming=streaming)
        
        # Map preprocessing
        return dataset.map(
            lambda example: {
                'pixel_values': processor(
                    images=example['img'],
                    return_tensors='pt',
                    padding=True
                )['pixel_values'][0],
                'text': str(example.get('fine_label', ''))  # Convert label to text
            },
            remove_columns=dataset.column_names
        ) 