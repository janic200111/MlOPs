#!/usr/bin/env python3
# chromadb_stats.py

import chromadb
from pathlib import Path

def show_chroma_stats(persist_dir: str, collection_name: str) -> None:
    """
    Open the ChromaDB collection stored under `persist_dir` and print basic statistics:
      - List of collections in that directory
      - Whether `collection_name` exists
      - Number of items in the collection
      - Embedding dimension (inferred from one sample entry)
      - Keys and a sample of the stored metadata and document text
    """
    # 1. Connect to the persistent Chroma client on disk
    client = chromadb.PersistentClient(path=persist_dir)

    # 2. List all collections under this directory
    collections = [col.name for col in client.list_collections()]
    print(f"Collections in '{persist_dir}': {collections!r}\n")

    # 3. Check if the requested collection exists
    if collection_name not in collections:
        print(f"Collection '{collection_name}' not found under '{persist_dir}'.")
        return

    # 4. Load the existing collection
    collection = client.get_collection(name=collection_name)

    # 5. Print the total number of items (embeddings) in this collection
    total_items = collection.count()
    print(f"Collection '{collection_name}' contains {total_items} item(s).\n")

    # 6. Peek at one entry (to infer embedding dimension, inspect metadata, document text)
    peeked = collection.peek()  # no arguments, returns lists of length ≥1
    embeddings_list = peeked.get("embeddings", [])
    if len(embeddings_list) > 0:
        first_embedding = embeddings_list[0]
        embedding_dim = len(first_embedding)
        print(f"Embedding dimension: {embedding_dim}\n")
    else:
        print("No embeddings found in the collection.\n")

    # 7. Show what metadata fields are stored and print a sample metadata
    metadatas_list = peeked.get("metadatas", [])
    if len(metadatas_list) > 0:
        metadata_keys = list(metadatas_list[0].keys())
        print(f"Metadata keys (stored for each item): {metadata_keys}\n")
        print(f"Sample metadata: {metadatas_list[0]}\n")
    else:
        print("No metadata found in the collection.\n")

    # 8. Show a snippet of the stored document text
    documents_list = peeked.get("documents", [])
    if len(documents_list) > 0:
        snippet = documents_list[0][:500].replace("\n", " ")
        print(f"Sample document text (first 500 chars):\n  {snippet!r}…\n")
    else:
        print("No document text found in the collection.\n")


if __name__ == "__main__":
    # Adjust these paths if your ChromaDB files live elsewhere
    PERSIST_DIR = "E:/MLOps/chroma_db"
    COLLECTION_NAME = "rag_collection"

    show_chroma_stats(PERSIST_DIR, COLLECTION_NAME)
