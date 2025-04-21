import faiss
import numpy as np
import json
from pathlib import Path
import requests

# Define paths to your index and metadata files
INDEX_PATH = "my_faiss_index.faiss"
METADATA_PATH = "my_metadata.json"
#embedding url
EMBEDDING_URL = "http://localhost:11434/api/embeddings"

def load_index(index_path):
    """Loads the FAISS index from the specified path."""
    index = faiss.read_index(index_path)
    return index

def load_metadata(metadata_path):
    """Loads the metadata from the specified JSON file."""
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata

def get_embedding(text: str, embedding_url: str) -> np.ndarray:
    """Gets the embedding for a given text using the Ollama API."""
    response = requests.post(
        embedding_url,
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )
    response.raise_for_status()
    return np.array(response.json()["embedding"], dtype=np.float32)

def search_index(index, metadata, query, top_k=5, embedding_url = EMBEDDING_URL):
    """
    Searches the FAISS index for the given query and returns the top_k results.

    Args:
        index: The FAISS index.
        metadata: The metadata associated with the index.
        query: The query string.
        top_k: The number of results to return.

    Returns:
        A list of dictionaries, where each dictionary contains the document name,
        chunk, and chunk ID of the top_k results.
    """
    query_embedding = get_embedding(query, embedding_url).reshape(1, -1)  # Reshape for FAISS
    D, I = index.search(query_embedding, top_k)  # D: distances, I: indices

    results = []
    for idx in I[0]:  # Iterate through the top_k indices
        results.append(metadata[idx])  # Retrieve metadata for each result
    return results

if __name__ == '__main__':
    # Load the index and metadata
    index = load_index(INDEX_PATH)
    metadata = load_metadata(METADATA_PATH)

    # Example query
    query = "how sap integrates with other systems?"  # Replace with your query
    top_k = 5  # Number of results to retrieve

    # Search the index
    results = search_index(index, metadata, query, top_k)

    # Print the results
    print(f"Query: {query}\n")
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Document: {result['doc_name']}")

        print(f"  Chunk: {result['chunk']}")
        print(f"  Chunk ID: {result['chunk_id']}")
        print("-" * 20)
    #print(index.search(np.array([[0.1]*768]), 5)) # Example to check if the index is working

##########################################################
import os
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import asyncio
from google import genai
#import google.generativeai as genai
from concurrent.futures import TimeoutError
from functools import partial
import json

# Load environment variables from .env file
load_dotenv()

# Access your API key and initialize Gemini client correctly
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents= f'''as a intelligent assistant provide answer to the query based 
    on the context provided in the below text. 
    \n" + {query} + "\n\n" + "Context: " + str({results})
    \n\n" + "Answer:"
    '''
)

print(f"Query: {query}\n")
print(f"\nGemini Response: {response.text}")