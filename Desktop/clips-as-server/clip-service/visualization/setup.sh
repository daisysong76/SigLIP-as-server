#!/bin/bash
# Setup script for the CLIP embedding visualization tools

# Create necessary directories
mkdir -p output/visualizations

# Install required packages
pip install scikit-learn umap-learn matplotlib requests tqdm

# Make the visualization script executable
chmod +x embedding_visualizer.py

echo "Setup complete. You can now run embedding_visualizer.py to generate and visualize embeddings." 