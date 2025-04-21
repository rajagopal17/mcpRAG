# mcpRAG
rag using Ollama as emebddings, gemini as LLM:

In this project i have created RAG using txt documents.
For embeddings - 'nomic embeddings' are created for txt using ollama locally
LLM  :  Gemini is used for LLM

All the txt files are chunked with file name, chunk id and chunk text in JSON format.
Each chunk is converted into embeddings and collected in a list

This embedding list is indexed using FAISS
when query is embedding using nomic embeddings, these embeddings are searched in FAISS index and relevant indices(location of chunk) is retrieved.  These indices are passed to JSON file to get the actual text.

THis text is passed to LLM with the query to formulate the answer.

