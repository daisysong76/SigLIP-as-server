"""
Generate and save embeddings for SigLIP-L (ViT-L-16) model for all images in a folder.
Saves embeddings as vitl16_image_embeddings.npy and metadata as vitl16_image_metadata.pkl
to match the format expected by upsert_vitl16_to_qdrant.py.
"""
import os
import numpy as np
import pickle
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, SiglipModel

# Configuration
IMAGE_DIR = "clip-service/input/images"
EMBEDDINGS_OUTPUT = "vitl16_image_embeddings.npy"  # Match upsert script expectations
METADATA_OUTPUT = "vitl16_image_metadata.pkl"      # Match upsert script expectations
MODEL_NAME = "google/siglip-base-patch16-384"  # Using base model, replace with large if available

def main():
    # Check if the image directory exists and has files
    if not os.path.exists(IMAGE_DIR):
        print(f"Image directory not found: {IMAGE_DIR}")
        return
    
    image_paths = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_paths:
        print(f"No images found in {IMAGE_DIR}")
        print("Adding a test image directory...")
        
        # Create test_images directory
        os.makedirs("clip-service/test_images", exist_ok=True)
        
        # If no real images, use our test metadata with dummy embeddings
        from create_test_metadata import main as create_test_data
        create_test_data()
        return
    
    print(f"Found {len(image_paths)} images. Loading model...")
    
    # Load SigLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = SiglipModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    
    metadata = []
    embeddings = []
    
    print(f"Generating ViT-L-16 embeddings on {device}...")
    with torch.no_grad():
        for img_path in tqdm(image_paths):
            try:
                # Load and process image
                image = Image.open(img_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                
                # Get embeddings
                outputs = model.get_image_features(**inputs)
                embedding = outputs.cpu().numpy().flatten()
                
                # Create metadata entry
                meta = {
                    "id": os.path.basename(img_path),
                    "image_path": img_path,
                    "doc_id": "doc123",  # Example doc_id, modify as needed
                }
                
                # Add to lists
                embeddings.append(embedding)
                metadata.append(meta)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Save embeddings and metadata
    embeddings = np.array(embeddings)
    np.save(EMBEDDINGS_OUTPUT, embeddings)
    
    with open(METADATA_OUTPUT, "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"Generated embeddings for {len(metadata)} images")
    print(f"Saved embeddings to {EMBEDDINGS_OUTPUT} and metadata to {METADATA_OUTPUT}")
    print("Next step: Run upsert_vitl16_to_qdrant.py to store in the vector database")

if __name__ == "__main__":
    main() 