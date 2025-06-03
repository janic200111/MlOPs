import os
import json
import logging
import torch
import chromadb
import boto3

from pathlib import Path
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── CONFIGURATION ───────────────────────────────

# 1. Paths for documents, Chroma persistence, and HF cache
DOCS_FOLDER       = "doc"                          # folder containing .txt files
CHROMA_PERSIST    = "E:/MLOps/chroma_db"           # where ChromaDB will save its index
HF_CACHE_DIR      = "E:/MLOps/huggingface-cache"   # local cache for HuggingFace models

# 2. Chroma collection configuration
CHROMA_COLLECTION = "rag_collection"

# 3. Embedding model settings
EMBEDDING_MODEL_NAME = "Salesforce/SFR-Embedding-Mistral"

# 4. Bedrock (instruct) model settings
BEDROCK_REGION    = "us-east-1"
BEDROCK_MODEL_ID  = "amazon.nova-micro-v1:0"
MAX_NEW_TOKENS    = 1000   # max tokens to sample from Bedrock

# 5. Retrieval settings
TOP_K_CHUNKS      = 3      # how many chunks to retrieve per query

# ── LOGGING SETUP ───────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ── FUNCTIONS ────────────────────────────────────────────────────────────────────

def check_system() -> torch.device:
    """
    Detect whether CUDA is available; return torch.device("cuda") or ("cpu").
    """
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        logger.info("CUDA not available. Using CPU.")
    return dev


def load_embedding_model(
    model_name: str,
    cache_dir: str,
    device: torch.device
) -> HuggingFaceEmbeddings:
    """
    Load a HuggingFace-based embedding model on the specified device,
    with the given cache directory.
    """
    logger.info(f"Loading embedding model '{model_name}' (cache: '{cache_dir}') onto {device}...")
    embedder = HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=cache_dir,
        model_kwargs={"device": device, "trust_remote_code": True},
    )
    logger.info(f"Embedding model loaded: {embedder.model_name}")
    return embedder


def load_documents_to_chunks(
    doc_folder: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> list[Document]:
    """
    Read all .txt files in `doc_folder`, wrap each in a Document,
    and split into chunks of up to chunk_size chars (with chunk_overlap).
    """
    logger.info(f"Loading documents from folder: '{doc_folder}'")
    docs: list[Document] = []
    for file in Path(doc_folder).glob("*.txt"):
        logger.info(f"  • Reading file: {file.name}")
        text = file.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": str(file)}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def get_or_build_chroma_index(
    chunks: list[Document],
    embedding_model: HuggingFaceEmbeddings,
    persist_dir: str,
    collection_name: str,
) -> Chroma:
    """
    If a Chroma collection named `collection_name` exists in `persist_dir`,
    load and return it; otherwise, build a new one from `chunks`.
    """
    logger.info(f"Initializing Chroma at '{persist_dir}' (collection: '{collection_name}')")
    client = chromadb.PersistentClient(path=persist_dir)
    existing = [col.name for col in client.list_collections()]

    if collection_name in existing:
        logger.info(f"Found existing collection '{collection_name}', loading it.")
        chroma_vs = Chroma(
            client=client,
            persist_directory=persist_dir,
            embedding_function=embedding_model,
            collection_name=collection_name,
        )
    else:
        logger.info(f"No existing collection found; building new collection '{collection_name}' (this may take a while)...")
        chroma_vs = Chroma.from_documents(
            client=client,
            documents=chunks,
            embedding=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_dir,
        )
        logger.info(f"Chroma index built and persisted to '{persist_dir}/{collection_name}'")

    return chroma_vs


def run_retrieval_qa_bedrock(
    chroma_vs: Chroma,
    question: str,
    top_k: int,
    max_new_tokens: int,
    region_name: str,
    model_id: str,
):
    """
    1) Retrieve top_k chunks for `question` from ChromaDB.
    2) Build a prompt combining those chunks with the question.
    3) Invoke the Bedrock instruct model via boto3.
    4) Print out retrieved snippets and the final generated answer.
    """
    logger.info(f"Retrieving top {top_k} chunks for question: '{question}'")
    retriever = chroma_vs.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(question)
    logger.info(f"Retrieved {len(docs)} chunks.")

    print("\n[CONTEXT CHUNKS]")
    for i, doc in enumerate(docs, start=1):
        snippet = doc.page_content.replace("\n", " ")[:200]
        print(f"{i}. Source: {doc.metadata['source']}\n   {snippet!r}…")
    print()

    # Build concatenated context + question prompt
    context_text = "\n\n---\n\n".join([d.page_content for d in docs])
    prompt = (
        "Answer the question based on the following context. "
        'If the answer is not contained in the context, respond with "I don\'t know."\n\n'
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
    )

    payload = {
        "inferenceConfig": {"max_new_tokens": max_new_tokens},
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
    }

    logger.info(f"Invoking Bedrock model '{model_id}' with max_new_tokens={max_new_tokens}")
    client = boto3.client("bedrock-runtime", region_name=region_name)
    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload).encode("utf-8"),
    )

    body_bytes = response["body"].read()
    obj = json.loads(body_bytes.decode("utf-8"))

    # Extract generated text
    try:
        generated_text = obj["output"]["message"]["content"][0]["text"].strip()
    except (KeyError, IndexError):
        logger.error("Could not extract generated text; full response follows:")
        logger.error(json.dumps(obj, indent=2))
        return

    print("\n[GENERATED ANSWER]")
    print(generated_text)
    print("==================\n")


def main():
    # ─── Step 1: Load & chunk documents ─────────────────────────────────────────
    chunks = load_documents_to_chunks(
        doc_folder=DOCS_FOLDER,
        chunk_size=1000,
        chunk_overlap=200
    )

    # ─── Step 2: Detect GPU/CPU, then load the embedding model ─────────────────
    device = check_system()
    embedder = load_embedding_model(
        model_name=EMBEDDING_MODEL_NAME,
        cache_dir=HF_CACHE_DIR,
        device=device,
    )

    # ─── Step 3: Build or load ChromaDB index ─────────────────────────────────
    chroma_vs = get_or_build_chroma_index(
        chunks=chunks,
        embedding_model=embedder,
        persist_dir=CHROMA_PERSIST,
        collection_name=CHROMA_COLLECTION,
    )

    # ─── Step 4: Rolling loop (ask / retrieve‐&‐generate) ───────────────────────
    print("\nEnter your question below. Type 'exit' or 'quit' to end.\n")
    while True:
        question = input("Your question (or 'exit'): ").strip()
        if question.lower() in {"exit", "quit"}:
            logger.info("Exit command received; terminating program.")
            print("Exiting. Goodbye!")
            break
        if not question:
            continue

        run_retrieval_qa_bedrock(
            chroma_vs=chroma_vs,
            question=question,
            top_k=TOP_K_CHUNKS,
            max_new_tokens=MAX_NEW_TOKENS,
            region_name=BEDROCK_REGION,
            model_id=BEDROCK_MODEL_ID,
        )


if __name__ == "__main__":
    main()
