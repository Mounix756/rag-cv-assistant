# Convention de Commits Git

Pour garder un historique clair et structuré dans ce projet RAG, nous suivons la spécification **Conventional Commits**.

## Structure d'un message

```text
<type>(<périmètre facultatif>): <description courte au présent>

[corps du message facultatif pour détailler le "pourquoi"]
```

## Types autorisés

* **`feat`** : Nouvelle fonctionnalité (ex: ajout de la recherche vectorielle, création de la base ChromaDB).
* **`fix`** : Correction d'un bug ou dysfonctionnement dans le code.
* **`docs`** : Modifications de la documentation uniquement (ex: `README.md`, `COMMIT.md`).
* **`refactor`** : Modification du code qui ne corrige pas de bug et n'ajoute pas de fonctionnalité (optimisation, réorganisation).
* **`chore`** : Tâches de maintenance, configuration de l'environnement ou gestion des dépendances (ex: `.gitignore`, `requirements.txt`).
* **`test`** : Ajout ou modification de tests unitaires/d'intégration.

## Exemples adaptés au projet RAG

* `chore: initialisation de la structure du projet et de l'environnement`
* `docs: ajout des règles de commit dans COMMIT.md`
* `feat(ingest): chargement et extraction des textes du PDF`
* `feat(vectorstore): indexation des chunks dans ChromaDB avec sentence-transformers`
* `fix(retrieval): correction du score de similarité lors du filtrage`