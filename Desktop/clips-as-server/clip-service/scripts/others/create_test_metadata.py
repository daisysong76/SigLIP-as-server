"""
Create a test metadata file with dummy embeddings and LLaVA results
for testing the viewer and pipeline components.
"""
import pickle
import numpy as np
import os

OUTPUT_PATH = "clip-service/scripts/all_image_metadata.pkl"

def main():
    # Create a directory for test images
    os.makedirs("clip-service/test_images", exist_ok=True)
    
    # Generate some dummy metadata
    metadata = []
    
    for i in range(5):
        # Generate dummy embeddings
        clip_emb = np.random.randn(512).tolist()
        siglip_b_emb = np.random.randn(768).tolist()
        siglip_l_emb = np.random.randn(1024).tolist()
        
        # Create an entry
        entry = {
            "id": f"test_image_{i}",
            "image_path": f"clip-service/test_images/test_image_{i}.jpg",
            "clip_embedding": clip_emb,
            "siglip_b_embedding": siglip_b_emb,
            "siglip_l_embedding": siglip_l_emb,
            "llava_caption": f"This is a test caption for image {i}. It shows a beautiful landscape with mountains and a lake.",
            "llava_scene": f"In this scene, there are several people hiking near the mountains. The weather appears to be sunny with some clouds in the sky. Image {i} has particularly vivid colors.",
            "llava_table": f"No tables or figures detected in image {i}.",
            "llava_qa": f"The main object in image {i} is a mountain range. What's unusual is the specific lighting pattern creating a rainbow effect over the valley.",
            "llava_explanation": f"The key elements in image {i} are: 1) A mountain range in the background, 2) A lake in the foreground, 3) Hikers for scale. These elements create a sense of grandeur and adventure."
        }
        metadata.append(entry)
    
    # Save the metadata
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"Created test metadata with {len(metadata)} entries at {OUTPUT_PATH}")
    print("You can now run view_llava_results.py to see the dummy LLaVA outputs")

if __name__ == "__main__":
    main() 