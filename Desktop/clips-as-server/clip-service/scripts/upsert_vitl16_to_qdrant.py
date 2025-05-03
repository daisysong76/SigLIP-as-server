"""
This script uploads embeddings to Qdrant vector database
Contains some LLaVA reasoning code but seems more focused on indexing
Mixes several different operations: loading embeddings, Qdrant operations, and some LLaVA inference
Not the primary LLaVA reasoning script - appears to be more for database operations

Vector Database Integration: This script uploads embeddings to Qdrant, a vector database designed for similarity search with multiple vector types (CLIP, SigLIP-B, SigLIP-L).
Multimodal Data Storage: It stores:
Multiple embedding types for each image
LLaVA-generated captions and descriptions
Associated metadata like document IDs, page numbers
Similarity Search: It demonstrates how to perform vector similarity searches with filtering, which is crucial for retrieving relevant visual information.
Workflow Integration: It shows a complete workflow pipeline with embedding generation, LLaVA reasoning, database uploading, and visualization.
"""
import numpy as np
import pickle
from qdrant_client import QdrantClient, models
import os
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from PIL import Image
import torch
from transformers import pipeline
from agent_commands import run_generate_embeddings, run_llava_batch, run_upsert_to_qdrant, run_visualization

# Config
EMBEDDINGS_PATH = "vitl16_image_embeddings.npy"  # Path to your .npy file
METADATA_PATH = "clip-service/scripts/all_image_metadata.pkl"      # Path to your .pkl file
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
COLLECTION_NAME = "multimodal_docs"
EMBEDDING_DIM = 1024  # Change if your embedding size is different

# Load embeddings and metadata
embeddings = np.load(EMBEDDINGS_PATH)
if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}. Please check the path.")
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

assert len(embeddings) == len(metadata), "Embeddings and metadata must have the same length!"

# Connect to Qdrant
client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

# Create collection if not exists
client.recreate_collection(
    collection_name="multimodal_docs",
    vectors_config={
        "clip": models.VectorParams(size=clip_dim, distance=models.Distance.COSINE),
        "siglip_b": models.VectorParams(size=siglip_b_dim, distance=models.Distance.COSINE),
        "siglip_l": models.VectorParams(size=siglip_l_dim, distance=models.Distance.COSINE),
    }
)

# Prepare points
points = []
for idx, meta in enumerate(metadata):
    client.upsert(
        collection_name="multimodal_docs",
        points=[
            models.PointStruct(
                id=meta["id"],
                vector={
                    "clip": meta["clip_embedding"],
                    "siglip_b": meta["siglip_b_embedding"],
                    "siglip_l": meta["siglip_l_embedding"],
                },
                payload=meta
            )
        ]
    )

print(f"Upserted {len(points)} points to Qdrant collection '{COLLECTION_NAME}' at {QDRANT_HOST}:{QDRANT_PORT}")

# Example: search for similar images with a filter on doc_id
query_embedding = np.random.rand(1024)  # Replace with your real query embedding

results = client.search(
    collection_name="multimodal_docs",
    query_vector={"siglip_l": query_embedding.tolist()},
    query_filter=models.Filter(
        must=[
            models.FieldCondition(key="doc_id", match=models.MatchValue(value="doc123")),
            # ... more filters
        ]
    ),
    with_payload=True,
    limit=10,
)
for hit in results:
    print(f"Score: {hit.score}, Image: {hit.payload.get('image_path')}, Caption: {hit.payload.get('llava_caption')}")

device = "cuda" if torch.cuda.is_available() else "cpu"
llava_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf").to(device)
llava_processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

def llava_infer(image, prompt):
    inputs = llava_processor(images=image, text=prompt, return_tensors="pt").to(device)
    output = llava_model.generate(**inputs, max_new_tokens=128)
    return llava_processor.decode(output[0], skip_special_tokens=True)

# Example for a single image:
image = Image.open("path/to/image.jpg")

results = {
    "caption": llava_infer(image, "Describe this image."),
    "scene_description": llava_infer(image, "What is happening in this image? Describe the scene in detail."),
    "table_summary": llava_infer(image, "If there is a table or figure in this image, summarize its content."),
    "visual_qa": llava_infer(image, "What is the main object in this image? What is unusual or interesting here?"),
    "explanation": llava_infer(image, "Explain the key elements and relationships in this image."),
}
print(results)

caption_prompt = "Describe this image."
scene_prompt = "What is happening in this image? Describe the scene in detail."
table_prompt = "If there is a table or figure in this image, summarize its content."
qa_prompt = "What is the main object in this image? What is unusual or interesting here?"
explain_prompt = "Explain the key elements and relationships in this image."

# Example usage:
# image = Image.open("path/to/image.jpg")
# caption = llava_caption(image)

all_results = []
for img_path in image_paths:
    image = Image.open(img_path)
    result = {
        "image_path": img_path,
        "caption": llava_infer(image, caption_prompt),
        "scene_description": llava_infer(image, scene_prompt),
        "table_summary": llava_infer(image, table_prompt),
        "visual_qa": llava_infer(image, qa_prompt),
        "explanation": llava_infer(image, explain_prompt),
    }
    all_results.append(result)

# New metadata structure
new_metadata = {
    "id": "unique_id",
    "image_path": "...",
    "clip_embedding": [...],
    "siglip_b_embedding": [...],
    "siglip_l_embedding": [...],
    "llava_caption": "...",
    "llava_scene": "...",
    "llava_table": "...",
    "llava_qa": "...",
    "llava_explanation": "...",
    "ocr_text": "...",
    "doc_id": "...",
    "page": 1,
    # ... any other metadata
}

run_generate_embeddings()
run_llava_batch()
run_upsert_to_qdrant()
run_visualization() 