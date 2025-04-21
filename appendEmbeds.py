import os
from pathlib import Path
import faiss
import numpy as np
import requests
import json
import time
import ollama

# -- CONFIG --
CHUNK_SIZE = 60
CHUNK_OVERLAP = 10
INDEX_PATH = "my_faiss_index.faiss"  # Define where to save the index
METADATA_PATH = "my_metadata.json"    # Define where to save the metadata

# -- HELPERS --

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i+size])
        if chunk:
            chunks.append(chunk)
    return chunks

def get_embedding(text: str) -> np.ndarray:
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )
    response.raise_for_status()
    return np.array(response.json()["embedding"], dtype=np.float32)

def load_existing_index(index_path, metadata_path):
    """Loads an existing FAISS index and its metadata."""
    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

def process_document(file_path):
    """Processes a single document, chunking it and generating embeddings."""
    all_chunks = []
    metadata = []
    file_path = Path(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        chunks = chunk_text(content)
        for idx, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            all_chunks.append(embedding)
            metadata.append({
                "doc_name": file_path.name,
                "chunk": chunk,
                "chunk_id": f"{file_path.stem}_{idx}"
            })
    print(f"Processing {file_path.name}...")
    time.sleep(1)  # Small delay between files
    return all_chunks, metadata

# -- LOAD DOCS & CHUNK --
# Load existing index and metadata
index, metadata = load_existing_index(INDEX_PATH, METADATA_PATH)

# Process the new document
new_file_path = "C:\\Users\\Srivalli\\mygitrepo\\mcpRAG\\sapfiles\\SAP_GENAI.txt"  # Replace with your new document path
new_chunks, new_metadata = process_document(new_file_path)

# -- APPEND TO FAISS INDEX --
new_embeddings = np.stack(new_chunks)
index.add(new_embeddings)

print(f"✅ Added {len(new_chunks)} new chunks to the index")

# -- UPDATE METADATA --
metadata.extend(new_metadata)

# -- SAVE FAISS INDEX --
faiss.write_index(index, INDEX_PATH)
print(f"✅ Saved updated FAISS index to {INDEX_PATH}")

# -- SAVE METADATA --
with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)
print(f"✅ Saved updated metadata to {METADATA_PATH}")