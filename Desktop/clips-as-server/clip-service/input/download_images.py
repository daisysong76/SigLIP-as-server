"""Download sample images for CLIP testing."""
import os
import requests
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# COCO 2017 validation set URLs
COCO_URL = "http://images.cocodataset.org/val2017/"
COCO_ANNOTATIONS = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# Sample image IDs from COCO val2017
SAMPLE_IMAGE_IDS = [
    "000000000139",  # Person
    "000000000285",  # Dog
    "000000000632",  # Car
    "000000000724",  # Building
    "000000000776",  # Food
    "000000000785",  # Nature
    "000000000802",  # Sports
    "000000000872",  # Technology
    "000000000885",  # Art
    "000000000886",  # Urban
]

def download_image(image_id: str, output_dir: Path):
    """Download a single image from COCO dataset."""
    filename = f"{image_id}.jpg"
    url = COCO_URL + filename
    output_path = output_dir / filename
    
    if output_path.exists():
        return
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")

def main():
    """Download sample images for testing."""
    output_dir = Path("input/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading sample images...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(tqdm(
            executor.map(
                lambda x: download_image(x, output_dir),
                SAMPLE_IMAGE_IDS
            ),
            total=len(SAMPLE_IMAGE_IDS)
        ))
    
    print(f"Downloaded {len(SAMPLE_IMAGE_IDS)} images to {output_dir}")
    
if __name__ == "__main__":
    main() 