#!/usr/bin/env python3
"""
Simple visualization script for LLaVA reasoning results.
"""

import json
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

def main():
    parser = argparse.ArgumentParser(description="Visualize LLaVA reasoning results")
    parser.add_argument("--input-file", type=str, required=True,
                      help="Path to JSON or pickle file with LLaVA reasoning data")
    parser.add_argument("--output-dir", type=str, default="reasoning_viz",
                      help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.input_file}")
    if args.input_file.endswith('.json'):
        with open(args.input_file, 'r') as f:
            data = json.load(f)
    elif args.input_file.endswith('.pkl'):
        with open(args.input_file, 'rb') as f:
            data = pickle.load(f)
    else:
        raise ValueError("Unsupported file format")
    
    print(f"Loaded {len(data)} entries")
    
    # Extract embeddings
    embedding_types = {}
    
    # Find all embedding types in first entry
    first_entry = data[0]
    for key in first_entry:
        if key.endswith('_img_embedding') or key.endswith('_txt_embedding'):
            embedding_types[key] = []
    
    print(f"Found {len(embedding_types)} embedding types: {list(embedding_types.keys())}")
    
    # Extract embeddings
    for entry in data:
        for embed_type in embedding_types:
            if embed_type in entry:
                if isinstance(entry[embed_type], list):
                    embedding_types[embed_type].append(np.array(entry[embed_type]))
                else:
                    embedding_types[embed_type].append(entry[embed_type])
    
    # Convert lists to numpy arrays
    for embed_type in embedding_types:
        embedding_types[embed_type] = np.array(embedding_types[embed_type])
        print(f"  {embed_type}: {embedding_types[embed_type].shape}")
    
    # Find LLaVA reasoning fields
    reasoning_fields = {}
    for key in first_entry:
        if key.startswith('llava_') and not key.endswith('_embedding'):
            reasoning_fields[key] = []
    
    print(f"Found {len(reasoning_fields)} reasoning fields: {list(reasoning_fields.keys())}")
    
    # Extract reasoning outputs
    for entry in data:
        for field in reasoning_fields:
            if field in entry:
                reasoning_fields[field].append(entry[field])
            else:
                reasoning_fields[field].append("")
    
    # Visualize embeddings with PCA
    print("Visualizing embeddings with PCA")
    for embed_type, embeds in embedding_types.items():
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeds)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
        
        # Add labels
        for i, (x, y) in enumerate(reduced):
            plt.annotate(str(i), (x, y), fontsize=8, alpha=0.7)
        
        plt.title(f'PCA Projection - {embed_type}')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(args.output_dir, f'pca_{embed_type}.png'), dpi=300)
        plt.close()
    
    # Create reasoning length table
    print("Creating reasoning length analysis")
    
    # Calculate lengths of reasoning outputs
    field_names = list(reasoning_fields.keys())
    num_samples = len(list(reasoning_fields.values())[0])
    
    # Create length matrix
    length_matrix = np.zeros((num_samples, len(field_names)))
    
    for i, field in enumerate(field_names):
        for j, output in enumerate(reasoning_fields[field]):
            length_matrix[j, i] = len(output) if not output.startswith("Error:") else 0
    
    # Save statistics
    stats = {
        "num_samples": len(data),
        "embedding_types": list(embedding_types.keys()),
        "reasoning_fields": list(reasoning_fields.keys()),
        "avg_reasoning_length": {field: float(np.mean([len(x) for x in values])) for field, values in reasoning_fields.items()},
        "embedding_stats": {
            embed_type: {
                "shape": embeds.shape,
                "mean": float(np.mean(embeds)),
                "std": float(np.std(embeds)),
                "min": float(np.min(embeds)),
                "max": float(np.max(embeds))
            } for embed_type, embeds in embedding_types.items()
        }
    }
    
    # Save stats as JSON
    with open(os.path.join(args.output_dir, 'reasoning_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    plt.imshow(length_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Output Length (chars)')
    plt.title('LLaVA Reasoning Output Lengths')
    plt.xlabel('Reasoning Type')
    plt.ylabel('Sample Index')
    plt.xticks(range(len(field_names)), field_names, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'reasoning_length_heatmap.png'), dpi=300)
    plt.close()
    
    print(f"Visualization complete! Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main() 