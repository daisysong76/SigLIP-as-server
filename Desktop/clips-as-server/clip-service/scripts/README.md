# Scripts Directory

This folder contains utility and pipeline scripts for data ingestion, embedding, and vector database operations.

## Scripts

### upsert_vitl16_to_qdrant.py
- **Purpose:**
  - Loads ViT-L-16-SigLIP-384 image embeddings and associated metadata.
  - Upserts them into a Qdrant collection for fast, hybrid, and multi-modal retrieval.
- **Usage:**
  - Make sure your `.npy` (embeddings) and `.pkl` (metadata) files are present.
  - Start your Qdrant instance (locally or in the cloud).
  - Run the script:
    ```sh
    python upsert_vitl16_to_qdrant.py
    ```
  - The script will create the collection if needed and upsert all points with full metadata.

---

Add new scripts here as your pipeline grows (e.g., for LLaVA, agent orchestration, etc.). 