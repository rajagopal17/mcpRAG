# mcpRAG
RAG- using opensource embeddings, opensource vector database and Gemini LLM
______________________________________________________________________________

In this project i have created RAG using txt documents:

Embeddings      : 'nomic embeddings' are used with ollama locally
LLM             :  gemini-2.0-flash 
Vector Database : FAISS

All the txt files are chunked with file name, chunk id and chunk text in JSON format and stored locally.
Each chunk is converted into embeddings and collected in a list

This embedding list is indexed using FAISS and stored locally.
when query is embedding using nomic embeddings, these embeddings are searched in FAISS index and relevant indices(location of chunk) is retrieved.  These indices are passed to JSON file to get the actual text.

THis text is passed to LLM with the query to formulate the answer.

Additional text is appended to the exiting index and queries are run on the updated index by loading the stored index and embedding file.

