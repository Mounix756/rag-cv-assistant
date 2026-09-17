from pathlib import Path
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = Path("data/cv.pdf")
VECTORSTORE_DIR = Path("vectorstore")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    if not PDF_PATH.exists():
        print(f"Error: '{PDF_PATH}' not found. Please place your PDF in the data/ directory.")
        sys.exit(1)

    # Extraction du texte du PDF
    print(f"Loading {PDF_PATH}...")
    loader = PyPDFLoader(str(PDF_PATH))
    docs = loader.load()

    # Découpage en blocs ajustés pour la recherche sémantique
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks from {len(docs)} page(s).")

    # Vectorisation et indexation dans ChromaDB
    print(f"Initializing embeddings ({EMBEDDING_MODEL})...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Saving vectorstore to '{VECTORSTORE_DIR}'...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR)
    )

    print("Ingestion complete.")


if __name__ == "__main__":
    main()