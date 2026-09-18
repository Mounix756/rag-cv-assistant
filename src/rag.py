import argparse
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

VECTORSTORE_DIR = Path("vectorstore")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL = "openai/gpt-oss-120b"


def format_docs(docs):
    return "\n\n".join(doc.page_content.strip() for doc in docs)


def main():
    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable is not set.")
        print("Please create a .env file containing GROQ_API_KEY=your_api_key")
        sys.exit(1)

    if not VECTORSTORE_DIR.exists():
        print(f"Error: Vectorstore directory '{VECTORSTORE_DIR}' not found.")
        print("Please run 'python src/ingest.py' first.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Query CV using RAG pipeline.")
    parser.add_argument(
        "query",
        nargs="?",
        default="Présente brièvement le profil et les compétences principales du candidat.",
        help="Question à poser sur le CV",
    )
    args = parser.parse_args()

    # Initialisation des embeddings multilingues et réouverture de la base vectorielle
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=str(VECTORSTORE_DIR), embedding_function=embeddings)

    # Récupération élargie aux 5 segments les plus pertinents
    retriever = db.as_retriever(search_kwargs={"k": 5})

    # Initialisation du LLM via l'API Groq
    llm = ChatGroq(model_name=GROQ_MODEL, temperature=0.2)

    template = """Tu es un assistant recrutement chargé de répondre aux questions sur un candidat à partir de son CV.
Réponds de manière professionnelle et concise en te basant uniquement sur le contexte ci-dessous.
Si l'information n'est pas contenue dans le contexte, réponds simplement que l'information n'est pas renseignée dans le CV.

Contexte extrait du CV:
{context}

Question:
{question}

Réponse:"""

    prompt = ChatPromptTemplate.from_template(template)

    # Extraction des chunks et génération de la réponse
    docs = retriever.invoke(args.query)
    context_text = format_docs(docs)

    chain = prompt | llm | StrOutputParser()

    print(f"Question : {args.query}\n")
    response = chain.invoke({"context": context_text, "question": args.query})

    print("Réponse :")
    print(response)


if __name__ == "__main__":
    main()