import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration des chemins et constantes
DATA_DIR = Path("data")
VECTORSTORE_DIR = Path("vectorstore")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL = "openai/gpt-oss-120b"

load_dotenv()

st.set_page_config(page_title="RAG CV Assistant", layout="centered")


def process_pdf(uploaded_file):
    """Sauvegarde le PDF téléversé, découpe le texte et reconstruit la base vectorielle."""
    DATA_DIR.mkdir(exist_ok=True)
    pdf_path = DATA_DIR / "cv.pdf"

    # Enregistrement du fichier sur le disque
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extraction et découpage du texte
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = text_splitter.split_documents(docs)

    # Récréation de la base vectorielle ChromaDB
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR)
    )

    # Réinitialisation du cache Streamlit pour recharger la nouvelle base
    st.cache_resource.clear()


@st.cache_resource
def load_rag_components():
    """Charge en cache les embeddings, le retriever et le LLM."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=str(VECTORSTORE_DIR), embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})

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

    return retriever, prompt, llm


def format_docs(docs):
    return "\n\n".join(doc.page_content.strip() for doc in docs)


# Vérifications de configuration
if not os.getenv("GROQ_API_KEY"):
    st.error("Clé GROQ_API_KEY introuvable. Veuillez vérifier votre fichier .env.")
    st.stop()

st.title("Assistant RAG - CV")

# Panneau latéral pour le téléversement
with st.sidebar:
    st.header("Gestion du document")
    uploaded_file = st.file_uploader("Téléverser un nouveau CV (PDF)", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Indexer le document"):
            with st.spinner("Traitement du PDF et génération des embeddings..."):
                process_pdf(uploaded_file)
                st.success("Document indexé avec succès !")
                st.session_state.messages = []  # Réinitialise la discussion

if not VECTORSTORE_DIR.exists():
    st.info("Veuillez téléverser un fichier PDF dans le panneau latéral pour commencer.")
    st.stop()

st.caption("Posez vos questions sur les expériences, projets et compétences du candidat.")

retriever, prompt, llm = load_rag_components()

# Gestion de l'historique des messages
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone d'interaction utilisateur
if user_query := st.chat_input("Ex: Quelles sont les compétences principales ?"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyse du CV en cours..."):
            docs = retriever.invoke(user_query)
            context_text = format_docs(docs)

            chain = prompt | llm | StrOutputParser()
            response = chain.invoke({"context": context_text, "question": user_query})

            st.markdown(response)

            with st.expander("Voir les extraits du CV consultés (Context)"):
                for idx, doc in enumerate(docs, start=1):
                    page = doc.metadata.get("page", 0) + 1
                    st.write(f"**Extrait {idx} (Page {page}) :**")
                    st.text(doc.page_content.strip())

    st.session_state.messages.append({"role": "assistant", "content": response})