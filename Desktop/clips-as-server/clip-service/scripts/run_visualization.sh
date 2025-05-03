#!/bin/bash
# Activate the virtual environment and run the embedding visualization script
source "$(dirname "$0")/../.venv/bin/activate"
python "$(dirname "$0")/visualize_embeddings.py" 