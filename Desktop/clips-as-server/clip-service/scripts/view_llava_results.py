"""
View LLaVA reasoning results for images.
Simple tool to visualize and verify LLaVA outputs from the metadata file.
"""
import pickle
import os
import sys
import argparse
from PIL import Image
import matplotlib.pyplot as plt

def print_result(label, text):
    """Print a labeled result with formatting"""
    print(f"\n{'-'*80}")
    print(f"{label}:")
    print(f"{'-'*80}")
    print(f"{text}")

def main():
    parser = argparse.ArgumentParser(description="View LLaVA results for images")
    parser.add_argument("--metadata", type=str, default="clip-service/scripts/all_image_metadata.pkl",
                        help="Path to metadata file with LLaVA results")
    parser.add_argument("--index", type=int, default=None,
                        help="Index of specific image to show (default: show all)")
    parser.add_argument("--image-key", type=str, default="image_path",
                        help="Key in metadata for image path")
    parser.add_argument("--show-images", action="store_true",
                        help="Display the images alongside results")
    parser.add_argument("--limit", type=int, default=5,
                        help="Maximum number of images to show (default: 5)")
    args = parser.parse_args()
    
    if not os.path.exists(args.metadata):
        print(f"Metadata file not found: {args.metadata}")
        return 1
    
    # Load metadata
    with open(args.metadata, "rb") as f:
        metadata = pickle.load(f)
    
    print(f"Loaded {len(metadata)} entries from {args.metadata}")
    
    # Determine which entries to show
    if args.index is not None:
        if args.index >= len(metadata):
            print(f"Error: Index {args.index} is out of range (max: {len(metadata)-1})")
            return 1
        entries = [metadata[args.index]]
    else:
        entries = metadata[:min(args.limit, len(metadata))]
    
    # Display entries
    for i, entry in enumerate(entries):
        print(f"\n{'='*100}")
        print(f"RESULT {i+1}/{len(entries)}")
        print(f"{'='*100}")
        
        if args.show_images and args.image_key in entry and os.path.exists(entry[args.image_key]):
            try:
                img = Image.open(entry[args.image_key])
                plt.figure(figsize=(10, 8))
                plt.imshow(img)
                plt.axis('off')
                plt.show()
            except Exception as e:
                print(f"Error displaying image: {e}")
        
        # Print all LLaVA results
        llava_keys = [k for k in entry.keys() if k.startswith("llava_")]
        
        if not llava_keys:
            print("No LLaVA results found for this entry. Has LLaVA processing been run?")
            continue
            
        for key in llava_keys:
            display_name = key.replace("llava_", "").upper()
            print_result(display_name, entry[key])
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 