import subprocess
import sys
import os

def run_generate_embeddings():
    """Run the embedding generation script."""
    result = subprocess.run([
        sys.executable, os.path.join(os.path.dirname(__file__), 'generate_embeddings.py')
    ], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"generate_embeddings.py failed: {result.stderr}")
    return result.stdout

def run_llava_batch():
    """Run the batch LLaVA reasoning script."""
    result = subprocess.run([
        sys.executable, os.path.join(os.path.dirname(__file__), 'run_llava_batch.py')
    ], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"run_llava_batch.py failed: {result.stderr}")
    return result.stdout

def run_upsert_to_qdrant():
    """Run the upsert to Qdrant script."""
    result = subprocess.run([
        sys.executable, os.path.join(os.path.dirname(__file__), 'upsert_vitl16_to_qdrant.py')
    ], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"upsert_vitl16_to_qdrant.py failed: {result.stderr}")
    return result.stdout

def run_visualization():
    """Run the embedding visualization script."""
    result = subprocess.run([
        sys.executable, os.path.join(os.path.dirname(__file__), 'visualize_embeddings.py')
    ], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"visualize_embeddings.py failed: {result.stderr}")
    return result.stdout 