# RAG CV Assistant

Un système RAG (*Retrieval-Augmented Generation*) à but pédagogique permettant de poser des questions à un LLM sur le contenu d'un CV au format PDF.

## Architecture
1. **Ingestion** : Lecture du PDF (`pypdf`), découpage en chunks (`LangChain`), vectorisation (`sentence-transformers`).
2. **Stockage** : Indexation des embeddings dans une base vectorielle locale (`ChromaDB`).
3. **Retrieval & Génération** : Recherche des chunks pertinents et génération de réponse enrichie via LLM.

## Installation
1. Cloner le projet et créer l'environnement virtuel :

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
2. Placer votre CV nommé `cv.pdf` dans le dossier `data/`.