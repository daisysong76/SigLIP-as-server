"""
Visualize CLIP, SigLIP-B, and SigLIP-L embeddings using UMAP and Plotly.
Loads all_image_metadata.pkl, projects each embedding type, and shows an interactive Plotly scatter plot.
"""
import pickle
import numpy as np
import umap
import plotly.express as px
import pandas as pd
import os
import sys

# Accept metadata path as a command-line argument
if len(sys.argv) > 1:
    METADATA_PATH = sys.argv[1]
else:
    METADATA_PATH = "clip-service/scripts/all_image_metadata.pkl"

if not os.path.exists(METADATA_PATH):
    print(f"Metadata file not found: {METADATA_PATH}. Please check the path or provide it as an argument.")
    sys.exit(1)

# Load metadata
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

# Extract embeddings and labels
clip_embs = np.array([m["clip_embedding"] for m in metadata])
siglip_b_embs = np.array([m["siglip_b_embedding"] for m in metadata])
siglip_l_embs = np.array([m["siglip_l_embedding"] for m in metadata])
labels = [m.get("label", m["id"]) for m in metadata]  # Use label if available, else id

# UMAP projection for each embedding type
umap_clip = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine').fit_transform(clip_embs)
umap_siglip_b = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine').fit_transform(siglip_b_embs)
umap_siglip_l = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine').fit_transform(siglip_l_embs)

# Combine for side-by-side visualization
n = len(clip_embs)
df = pd.DataFrame({
    "x": np.concatenate([umap_clip[:,0], umap_siglip_b[:,0], umap_siglip_l[:,0]]),
    "y": np.concatenate([umap_clip[:,1], umap_siglip_b[:,1], umap_siglip_l[:,1]]),
    "model": (["CLIP"]*n + ["SigLIP-B"]*n + ["SigLIP-L"]*n),
    "label": labels*3
})

fig = px.scatter(
    df, x="x", y="y", color="model", hover_data=["label"],
    title="UMAP Projection of CLIP, SigLIP-B, and SigLIP-L Embeddings"
)
fig.show() 