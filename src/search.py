import argparse
from pathlib import Path
import sys

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

VECTORSTORE_DIR = Path("vectorstore")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main():
    parser = argparse.ArgumentParser(description="Query local vector database.")
    parser.add_argument("query", nargs="?", default="Quelles sont les compétences principales ?", help="Search query")
    parser.add_argument("-k", type=int, default=3, help="Number of results to retrieve")
    args = parser.parse_args()

    if not VECTORSTORE_DIR.exists():
        print(f"Error: Database directory '{VECTORSTORE_DIR}' not found.")
        print("Please run 'python src/ingest.py' first.")
        sys.exit(1)

    # Chargement du modèle d'embeddings et réouverture de la base
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=str(VECTORSTORE_DIR), embedding_function=embeddings)

    # Execution de la recherche par similarité (recherche L2 / cosinus)
    results = db.similarity_search_with_score(args.query, k=args.k)

    print(f"Query: '{args.query}'")
    print(f"Retrieved {len(results)} chunk(s):\n" + "-" * 50)

    for idx, (doc, score) in enumerate(results, start=1):
        page = doc.metadata.get("page", 0) + 1  # Numérotation base 1 pour l'affichage
        print(f"[{idx}] Score (distance): {score:.4f} | Page: {page}")
        print(doc.page_content.strip())
        print("-" * 50)


if __name__ == "__main__":
    main()