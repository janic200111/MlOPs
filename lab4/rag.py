from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import torch
import chromadb
import boto3
import json


def check_system():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def load_model(model_name, cache_dir, device):
    model = HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=cache_dir,
        model_kwargs={"device": device, "trust_remote_code": True},
    )

    print(f"Model loaded: {model.model_name}")
    return model


def load_documents_to_chunks(doc_folder="doc"):
    docs = []
    for file in Path(doc_folder).glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            print(f"Loading {file.name}")
            docs.append(Document(page_content=f.read(), metadata={"source": str(file)}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(docs)

    print(f"Total number of chunks created: {len(texts)}")
    return texts


def get_or_build_chroma_index(
    chunks,
    embedding_model,
    collection_name,
    persist_dir,
):

    client = chromadb.PersistentClient(path=persist_dir)
    existing = [col.name for col in client.list_collections()]

    if collection_name in existing:
        print(f"Loading existing Chroma collection: {collection_name}")
        chroma_vs = Chroma(
            client=client,
            persist_directory=persist_dir,
            embedding_function=embedding_model,
            collection_name=collection_name,
        )
    else:
        print(f"Creating new Chroma collection: {collection_name}")
        chroma_vs = Chroma.from_documents(
            client=client,
            documents=chunks,
            embedding=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_dir,
        )
        print(f"Chroma index built & persisted to: {persist_dir}/{collection_name}")

    return chroma_vs


def run_retrieval_qa_bedrock(
    chroma_vs,
    question,
    top_k=3,
    max_new_tokens=1000,
    region_name="us-east-1",
    model_id="amazon.nova-micro-v1:0",
):

    retriever = chroma_vs.as_retriever(search_kwargs={"k": top_k})
    results = retriever.get_relevant_documents(question)

    print(f"Query: {question}")
    print(f"Top {top_k} results:")
    for i, doc in enumerate(results):
        print(f"{i + 1}. {doc.page_content[:200]}... (Source: {doc.metadata['source']})")

    context = "\n".join([doc.page_content for doc in results])

    prompt_text = (
        "Answer the question based on the following context. "
        'If the answer is not contained in the context, respond with "I don\'t know."\n\n'
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    payload = {
        "inferenceConfig": {"max_new_tokens": max_new_tokens},
        "messages": [{"role": "user", "content": [{"text": prompt_text}]}],
    }

    print(f"Invoking Bedrock model: {model_id}\n")
    runtime_client = boto3.client("bedrock-runtime", region_name=region_name)
    response = runtime_client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload).encode("utf-8"),
    )

    body_bytes = response["body"].read()
    obj = json.loads(body_bytes.decode("utf-8"))

    try:
        generated_text = obj["output"]["message"]["content"][0]["text"].strip()
    except (KeyError, IndexError):
        print("ERROR: Could not extract generated text. Full response:")
        print(json.dumps(obj, indent=2))
        return

    print("\n=== Generated Answer ===")
    print(generated_text)
    print("\n=== End of Answer ===\n")


def main():
    chunks = load_documents_to_chunks()
    device = check_system()
    model = load_model(
        model_name="Salesforce/SFR-Embedding-Mistral",
        device=device,
        cache_dir="E:/MLOps/huggingface-cache",
    )
    chroma_vs = get_or_build_chroma_index(
        chunks=chunks,
        embedding_model=model,
        collection_name="rag_collection",
        persist_dir="E:/MLOps/chroma_db",
    )

    print("\nEnter your question below. Type 'exit' to end.\n")
    while True:
        question = input("Your question (or 'exit'): ").strip()
        if question.lower() in {"exit"}:
            print("Exiting. Goodbye!")
            break
        if not question:
            continue

        run_retrieval_qa_bedrock(
            chroma_vs=chroma_vs,
            question=question,
            model_id="amazon.nova-micro-v1:0",
            max_new_tokens=1000,
            top_k=3,
        )


if __name__ == "__main__":
    main()
