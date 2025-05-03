# CLIP Embedding Visualization Tools

This directory contains tools for visualizing CLIP embeddings, performing nearest neighbor searches, and generating visualizations.

## Features

- Save embeddings and their corresponding IDs
- Visualize embeddings using t-SNE and UMAP dimensionality reduction
- Perform and visualize top-K nearest neighbor searches (text-to-image and image-to-text)
- Generate beautiful visualizations of the embedding space

## Setup

1. Run the setup script to install dependencies:

```bash
./setup.sh
```

2. Make sure you have the required Python packages:
   - scikit-learn (for t-SNE)
   - umap-learn (for UMAP)
   - matplotlib (for visualization)
   - requests (for downloading images)
   - tqdm (for progress bars)

## Usage

### Basic Usage

Generate embeddings and create a t-SNE visualization:

```bash
python embedding_visualizer.py --tsne
```

### Command Line Arguments

- `--model`: CLIP model name (default: "ViT-B/32")
- `--output-dir`: Output directory (default: "output/visualizations")
- `--dataset`: Dataset to use (default: "nlphuji/flickr30k")
- `--max-samples`: Maximum samples to process (default: 500)
- `--save-file`: Filename to save embeddings (default: "embeddings.pkl")
- `--load`: Load embeddings instead of generating
- `--tsne`: Generate t-SNE visualization
- `--umap`: Generate UMAP visualization
- `--text-query`: Text query for image search
- `--image-query`: Image path for text search
- `--top-k`: Number of top results to show (default: 5)

### Examples

1. Generate embeddings and save them:

```bash
python embedding_visualizer.py
```

2. Load saved embeddings and create a t-SNE visualization:

```bash
python embedding_visualizer.py --load --tsne
```

3. Load saved embeddings and search for images matching a text query:

```bash
python embedding_visualizer.py --load --text-query "a dog running on the beach" --top-k 10
```

4. Load saved embeddings and search for text matching an image:

```bash
python embedding_visualizer.py --load --image-query "path/to/image.jpg" --top-k 5
```

5. Generate embeddings with a larger dataset and create both t-SNE and UMAP visualizations:

```bash
python embedding_visualizer.py --max-samples 1000 --tsne --umap
```

## Output

All outputs are saved to the specified output directory (default: "output/visualizations"):

- Embeddings are saved as pickle files
- Visualizations are saved as PNG files
- Search results are displayed and can also be saved as images

## Advanced Usage

You can also use the `EmbeddingVisualizer` class in your own code:

```python
from embedding_visualizer import EmbeddingVisualizer
import asyncio

async def main():
    visualizer = EmbeddingVisualizer(model_name="ViT-B/32")
    
    # Load existing embeddings
    visualizer.load_embeddings("embeddings.pkl")
    
    # Generate t-SNE visualization
    visualizer.visualize_tsne()
    
    # Search for images matching text
    visualizer.visualize_text_to_image_search("a cat sitting on a window", k=5)

# Run the async function
asyncio.run(main())
``` 