"""
Visualize and evaluate SigLIP-L (ViT-L-16) embeddings.
- UMAP+Plotly for interactive 2D visualization.
- Computes Recall@1, Recall@5, and MRR for retrieval quality.
- Designed for best-practice, automated embedding evaluation (2025).
"""
import pickle
import numpy as np
import umap
import plotly.express as px
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
import sys
import os

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

# Extract SigLIP-L embeddings and labels
siglip_l_embs = np.array([m["siglip_l_embedding"] for m in metadata])
labels = [m.get("label", m["id"]) for m in metadata]

# --- Visualization ---
umap_proj = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine').fit_transform(siglip_l_embs)
df = pd.DataFrame({
    "x": umap_proj[:,0],
    "y": umap_proj[:,1],
    "label": labels
})
fig = px.scatter(
    df, x="x", y="y", color="label", hover_data=["label"],
    title="UMAP Projection of SigLIP-L (ViT-L-16) Embeddings"
)
fig.show()

# --- Retrieval Metrics ---
def recall_at_k(sim_matrix, ground_truth, k=1):
    top_k = np.argsort(-sim_matrix, axis=1)[:, :k]
    hits = [gt in top_k[i] for i, gt in enumerate(ground_truth)]
    return np.mean(hits)

def mean_reciprocal_rank(sim_matrix, ground_truth):
    ranks = []
    for i, gt in enumerate(ground_truth):
        ranking = np.argsort(-sim_matrix[i])
        rank = np.where(ranking == gt)[0][0] + 1
        ranks.append(1.0 / rank)
    return np.mean(ranks)

sim_matrix = 1 - cosine_distances(siglip_l_embs, siglip_l_embs)
ground_truth = np.arange(len(labels))
recall1 = recall_at_k(sim_matrix, ground_truth, k=1)
recall5 = recall_at_k(sim_matrix, ground_truth, k=5)
mrr = mean_reciprocal_rank(sim_matrix, ground_truth)

print(f"Retrieval Metrics for SigLIP-L (ViT-L-16) embeddings:")
print(f"  Recall@1:  {recall1:.3f}")
print(f"  Recall@5:  {recall5:.3f}")
print(f"  MRR:       {mrr:.3f}") 