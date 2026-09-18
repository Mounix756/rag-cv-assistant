# RAG CV Assistant

Un système RAG (*Retrieval-Augmented Generation*) à but pédagogique permettant de poser des questions à un LLM sur le contenu d'un CV au format PDF.

## Architecture & Fonctionnement
1. **Ingestion** : Chargement du PDF (`PyPDFLoader`), découpage dynamique en segments (`RecursiveCharacterTextSplitter`).
2. **Vectorisation & Stockage** : Transformation du texte en vecteurs sémantiques (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) et stockage local dans `ChromaDB`.
3. **Retrieval** : Extraction des $k=5$ morceaux du CV les plus pertinents par calcul de similarité vectorielle.
4. **Génération** : Injection du contexte dans un prompt strict et génération de la réponse via l'API Groq (`openai/gpt-oss-120b`).

## Limites Techniques (v1)
* **Parsing PDF basique (`PyPDFLoader`)** : Absence d'OCR (incapable de lire les CV sous forme d'images ou de scans) et risque de mélange de texte sur les mises en page à deux colonnes.
* **Absence d'agrégation globale** : La recherche vectorielle par similarité ne sait pas calculer des critères cumulatifs sur l'ensemble du document (ex : *"calculer le nombre total d'années d'expérience"*).
* **Analyse mono-document** : Conçu pour interroger un seul CV à la fois, le système ne permet pas le tri, la comparaison ni le filtrage de masse sur un lot de candidats.
* **Découpage fixe (Chunking)** : Le découpage arbitraire par taille (`chunk_size=500`) peut parfois isoler une compétence de son contexte ou couper une phrase au milieu d'une section.

## Structure du Projet

```text
rag-cv-assistant/
├── data/               # Dossier contenant le CV source (cv.pdf)
├── vectorstore/        # Base vectorielle locale ChromaDB (générée automatiquement)
├── src/
│   ├── ingest.py       # Script d'extraction, de chunking et de vectorisation du PDF
│   ├── search.py       # Script de test rapide de la recherche par similarité (sans LLM)
│   └── rag.py          # Pipeline RAG complet en ligne de commande
├── .env.example        # Modèle de variables d'environnement
├── .gitignore          # Exclusion des fichiers temporaires, données sensibles et virtuenvs
├── app.py              # Application web Streamlit avec module de téléversement
├── COMMIT.md           # Conventions de nommage des commits Git
├── README.md           # Documentation du projet
└── requirements.txt    # Dépendances Python
```

## Prérequis & Installation

### 1. Clonage du dépôt et création de l'environnement Conda
```bash
git clone [https://github.com/votre-utilisateur/rag-cv-assistant.git](https://github.com/votre-utilisateur/rag-cv-assistant.git)
cd rag-cv-assistant

# Création et activation de l'environnement Conda avec Python 3.10
conda create -n llm_env python=3.10 -y
conda activate llm_env
```

### 2. Installation des dépendances
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configuration des variables d'environnement
Créez un fichier `.env` à la racine du projet à partir du modèle `.env.example` :
```bash
cp .env.example .env
```
Ajoutez-y votre clé d'API Groq (obtenue sur [console.groq.com](https://console.groq.com/)) :
```env
GROQ_API_KEY=gsk_votre_cle_api_groq_ici
```

## Utilisation

### Étape 1 : Placer le CV source et exécuter l'ingestion
Déposez votre fichier CV nommé `cv.pdf` dans le dossier `data/`, puis exécutez le script d'ingestion pour construire la base vectorielle :
```bash
python src/ingest.py
```

### Étape 2 (Optionnel) : Tester la recherche vectorielle seule
Pour vérifier la pertinence des chunks extraits sans faire appel au LLM :
```bash
python src/search.py "Quelles sont les compétences en Python et Flutter ?"
```

### Étape 3 : Poser des questions via la CLI RAG
Interrogez le système RAG directement depuis le terminal :
```bash
python src/rag.py "Quelles sont les expériences en développement mobile ?"
```

### Étape 4 : Lancer l'application Web (Streamlit)
Démarrez l'interface graphique interactive pour discuter avec le CV ou téléverser un nouveau document :
```bash
streamlit run app.py
```

## Stack Technique

* **Orchestration RAG :** LangChain (`langchain-chroma`, `langchain-groq`, `langchain-huggingface`)
* **Modèle d'Embeddings :** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
* **Base Vectorielle :** ChromaDB
* **Moteur LLM :** Groq API (`openai/gpt-oss-120b`)
* **Interface Utilisateur :** Streamlit