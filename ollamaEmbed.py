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

# -- LOAD DOCS & CHUNK --




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

file_path = Path("C:\\Users\\Srivalli\\mygitrepo\\mcpRAG\\sapfiles\\SAP_Joule.txt")
all_chunks, metadata =process_document(file_path)




# -- CREATE FAISS INDEX --
dimension = len(all_chunks[0])  # Should be 768 for nomic-embed-text
index = faiss.IndexFlatL2(dimension)
index.add(np.stack(all_chunks))

print(f"✅ Indexed {len(all_chunks)} chunks" )

# -- SAVE FAISS INDEX --
faiss.write_index(index, INDEX_PATH)
print(f"✅ Saved FAISS index to {INDEX_PATH}")

# -- SAVE METADATA --
with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)
print(f"✅ Saved metadata to {METADATA_PATH}")


