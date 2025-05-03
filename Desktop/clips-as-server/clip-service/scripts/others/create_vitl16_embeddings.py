"""
Create a vitl16_image_embeddings.npy file from existing metadata.
This is for testing the upsert_vitl16_to_qdrant.py script without
needing to run the full embedding generation pipeline.
"""
import pickle
import numpy as np
import os

# Paths
METADATA_PATH = "clip-service/scripts/all_image_metadata.pkl"
EMBEDDINGS_OUTPUT = "vitl16_image_embeddings.npy"

def main():
    # Check if metadata exists
    if not os.path.exists(METADATA_PATH):
        print(f"Metadata file not found: {METADATA_PATH}")
        return
    
    # Load metadata
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    
    print(f"Loaded metadata with {len(metadata)} entries")
    
    # Extract embeddings - use SigLIP-L embeddings if available, otherwise create random ones
    embeddings = []
    for entry in metadata:
        if "siglip_l_embedding" in entry:
            # Convert to numpy array if it's a list
            emb = np.array(entry["siglip_l_embedding"])
            embeddings.append(emb)
        else:
            # Create random embedding of expected dimension (1024 for ViT-L-16)
            emb = np.random.randn(1024)
            embeddings.append(emb)
    
    # Convert to numpy array
    embeddings = np.array(embeddings)
    
    # Save embeddings
    np.save(EMBEDDINGS_OUTPUT, embeddings)
    
    print(f"Created {EMBEDDINGS_OUTPUT} with shape {embeddings.shape}")
    print(f"Next step: Run upsert_vitl16_to_qdrant.py to store in Qdrant")

if __name__ == "__main__":
    main() 