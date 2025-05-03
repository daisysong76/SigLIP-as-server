"""
The main script for LLaVA reasoning is run_llava_batch.py
Batch LLaVA visual reasoning for all images. Generates captions, scene descriptions, table/figure summaries, visual Q&A, and explanations.
Saves results in metadata for each image for downstream retrieval and agent use.
This is the main script for running LLaVA reasoning in batch mode
Uses the LLaVAProcessor from llava_utils.py
Takes a metadata pickle file as input
Processes all images with various prompt templates
Saves the enriched metadata with LLaVA reasoning outputs
Primary script for applying LLaVA reasoning to a collection of images
"""
import pickle
import os
import sys
from tqdm import tqdm
import argparse
from llava_utils import LLaVAProcessor, PROMPT_TEMPLATES

def main():
    parser = argparse.ArgumentParser(description="Batch LLaVA reasoning on images")
    parser.add_argument("--metadata", type=str, default="clip-service/scripts/all_image_metadata.pkl",
                        help="Path to metadata pickle file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (defaults to overwriting input file)")
    parser.add_argument("--model", type=str, default="llava-hf/llava-1.5-7b-hf",
                        help="LLaVA model name")
    parser.add_argument("--image-key", type=str, default="image_path",
                        help="Key in metadata containing image path")
    parser.add_argument("--skip-errors", action="store_true",
                        help="Skip errors and continue processing")
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.metadata
    
    if not os.path.exists(args.metadata):
        print(f"Metadata file not found: {args.metadata}")
        return 1
    
    # Load metadata
    try:
        with open(args.metadata, "rb") as f:
            metadata = pickle.load(f)
        
        print(f"Loaded {len(metadata)} entries from {args.metadata}")
    except Exception as e:
        print(f"Error loading metadata file: {e}")
        return 1
    
    # Initialize LLaVA
    try:
        print(f"Initializing LLaVA model: {args.model}")
        llava = LLaVAProcessor(model_name=args.model)
    except Exception as e:
        print(f"Error initializing LLaVA model: {e}")
        return 1
    
    # Process all images with LLaVA
    print("Running LLaVA inference (this may take a while)...")
    
    # Default prompts from PROMPT_TEMPLATES for all standard reasoning tasks
    prompts = {
        "llava_caption": PROMPT_TEMPLATES["caption"],
        "llava_scene": PROMPT_TEMPLATES["scene"], 
        "llava_table": PROMPT_TEMPLATES["table"],
        "llava_qa": PROMPT_TEMPLATES["qa"],
        "llava_explanation": PROMPT_TEMPLATES["explanation"]
    }
    
    # Check if the image_key exists in the metadata
    sample_entry = metadata[0] if metadata else {}
    if args.image_key not in sample_entry:
        print(f"Warning: '{args.image_key}' not found in metadata. Available keys: {list(sample_entry.keys())}")
        if args.skip_errors:
            print("Continuing with empty images due to --skip-errors flag")
        else:
            print("Use --skip-errors to continue with empty images")
            return 1
    
    # Try to batch process all images
    try:
        metadata = llava.batch_process(
            metadata=metadata,
            prompts=prompts,
            image_key=args.image_key,
            show_progress=True
        )
    except Exception as e:
        print(f"Error during batch processing: {e}")
        if not args.skip_errors:
            return 1
        print("Continuing due to --skip-errors flag")
    
    # Save enriched metadata
    try:
        print(f"Saving enriched metadata to {args.output}")
        with open(args.output, "wb") as f:
            pickle.dump(metadata, f)
        
        # Also save a JSON version for easier inspection
        json_output = args.output.replace(".pkl", ".json")
        if json_output != args.output:
            import json
            
            # Convert numpy arrays to lists for JSON serialization
            json_metadata = []
            for entry in metadata:
                json_entry = {}
                for k, v in entry.items():
                    if isinstance(v, list) and v and isinstance(v[0], float):
                        # Skip large embedding arrays in JSON output
                        json_entry[k] = f"[{len(v)} float values]"
                    else:
                        json_entry[k] = v
                json_metadata.append(json_entry)
                
            with open(json_output, "w") as f:
                json.dump(json_metadata, f, indent=2)
            print(f"Also saved JSON version to {json_output}")
    
    except Exception as e:
        print(f"Error saving results: {e}")
        return 1
    
    print(f"LLaVA processing complete. Saved enriched metadata to {args.output}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 