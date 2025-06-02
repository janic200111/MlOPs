from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import torch
import chromadb


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
    # print("First chunk preview:")
    # print(texts[0].page_content[:500])

    return texts


def build_chroma_index(
    chunks,
    embedding_model,
    collection_name,
    persist_dir,
    rebuild=False,
):

    client = chromadb.PersistentClient(path=persist_dir)

    if rebuild:
        try:
            client.delete_collection(collection_name)
        except Exception as e:
            print(f"Collection {collection_name} does not exist or could not be deleted: {e}")

        chroma_vs = Chroma.from_documents(
            client=client,
            documents=chunks,
            embedding=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_dir,
        )

    else:
        chroma_vs = Chroma(
            client=client,
            embedding_function=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_dir,
        )

    print(f"Chroma index built & persisted to: {persist_dir}/{collection_name}")
    return chroma_vs


def main():
    chunks = load_documents_to_chunks()
    device = check_system()
    model = load_model(
        model_name="Salesforce/SFR-Embedding-Mistral",
        device=device,
        cache_dir="E:/MLOps/huggingface-cache",
    )
    chroma_vs = build_chroma_index(
        chunks=chunks,
        embedding_model=model,
        collection_name="rag_collection",
        persist_dir="E:/MLOps/chroma_db",
        rebuild=False,
    )


if __name__ == "__main__":
    main()
