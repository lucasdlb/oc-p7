# Agenda Culturel 13 — Assistant RAG

Assistant de recommandation d'événements culturels pour les Bouches-du-Rhône (département 13), basé sur une architecture RAG (Retrieval-Augmented Generation).

---

## Sommaire

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture](#architecture)
3. [Pipeline de données](#pipeline-de-données)
4. [API REST](#api-rest)
5. [Interface utilisateur](#interface-utilisateur)
6. [Configuration](#configuration)
7. [Installation et démarrage](#installation-et-démarrage)
8. [Tests](#tests)
9. [Docker](#docker)
10. [Choix architecturaux et alternatives envisageables](#choix-architecturaux-et-alternatives-envisageables)

---

## Vue d'ensemble

Le projet répond à la question : **"Quels événements culturels se passent près de chez moi dans les Bouches-du-Rhône ?"**

L'utilisateur pose une question en langage naturel. Le système récupère les événements les plus pertinents depuis un index vectoriel, puis les transmet à un LLM (Mistral) pour générer une réponse contextualisée en français.

```
[OpenDataSoft] → fetch → clean → embed → [FAISS]
                                              ↓
                         [Utilisateur] → /ask → [RAG chain] → réponse
```

---

## Architecture

```
oc-p7/
├── scripts/
│   ├── fetch_events.py      # Récupération OpenDataSoft (pagination)
│   ├── clean_events.py      # Nettoyage et normalisation des données
│   └── build_index.py       # Chunking, embeddings Mistral, index FAISS
├── api/
│   ├── main.py              # FastAPI (endpoints /health, /ask, /rebuild)
│   ├── rag.py               # Chaîne RAG (RetrievalQA + Mistral)
│   └── models.py            # Schémas Pydantic (requêtes/réponses)
├── ui/
│   └── app.py               # Interface Gradio (chatbot + santé + rebuild)
├── tests/
│   ├── test_fetch.py
│   ├── test_clean.py
│   ├── test_index.py
│   ├── test_api.py
│   └── integration/
│       └── evaluate_rag.py  # Évaluation Ragas
├── docs/
│   └── test_set.json        # 12 paires Q/R annotées pour l'évaluation
├── config.py                # Chargement config + validation Pydantic
├── config.toml              # Config statique (production)
├── debug.toml               # Config statique (debug, données réduites)
└── logging_config.py        # Logging structuré (JSON prod / texte debug)
```

### Flux de données

```
OpenDataSoft API (public)
        │  HTTP GET paginé
        ▼
fetch_events.py  →  data/events_raw.json
        │
clean_events.py  →  data/events_clean.csv
        │
build_index.py   →  vector_store/ (FAISS + embeddings Mistral)
        │
  api/rag.py     ←  FAISS.load_local()
        │
  api/main.py    →  /ask  →  RetrievalQA (k=4) → mistral-large-latest
        │
    ui/app.py    →  chatbot Gradio
```

---

## Pipeline de données

### 1. Récupération (`scripts/fetch_events.py`)

- **Source** : API publique OpenDataSoft — jeu de données `evenements-publics-openagenda`
- **Filtre** : département Bouches-du-Rhône (`location_department` ou code postal `13*`) + événements futurs (`lastdate_end > now()`)
- **Pagination** : 100 événements par page, `time.sleep(0.3)` entre les appels (rate-limit)
- **Double filtrage** : côté API (OQL) et côté Python via `is_future_event()` qui parse le champ `timings` (JSON string) pour vérifier les créneaux réels
- **Sortie** : `data/events_raw.json`

### 2. Nettoyage (`scripts/clean_events.py`)

- **Validation** : présence obligatoire de `title_fr`, `description_fr`, `timings`, `uid`
- **Normalisation** : renommage des champs OpenDataSoft vers un schéma interne (12 colonnes)
- **Nettoyage HTML** : `BeautifulSoup` pour extraire le texte des descriptions enrichies
- **Traçabilité** : les événements rejetés sont sauvegardés dans `data/events_skipped.json` avec la raison
- **Sortie** : `data/events_clean.csv`

### 3. Vectorisation (`scripts/build_index.py`)

- **Document** : `"{title}. {description}"` avec métadonnées (`uid`, `city`, `firstdate_begin`, `lastdate_end`)
- **Chunking** : `RecursiveCharacterTextSplitter` — 512 tokens, chevauchement 50 (préserve les métadonnées sur tous les chunks)
- **Embeddings** : `MistralAIEmbeddings(model="mistral-embed")` — vecteurs de dimension 1024
- **Index** : FAISS construit par **lots de 250 chunks** avec backoff exponentiel (3 tentatives) pour absorber les limites de l'API
- **Sortie** : `vector_store/` (fichiers `index.faiss` + `index.pkl`)

---

## API REST

Démarrée avec `uvicorn api.main:app`.

| Méthode | Endpoint   | Description |
|---------|------------|-------------|
| `GET`   | `/health`  | Statut de l'API + état de l'index |
| `POST`  | `/ask`     | Pose une question, retourne réponse + sources |
| `POST`  | `/rebuild` | Relance le pipeline complet (fetch → clean → index) |

### Exemple `/ask`

```json
// POST /ask
{ "question": "Quels festivals de jazz ont lieu à Aix en juillet ?" }

// Réponse
{
  "question": "Quels festivals de jazz ont lieu à Aix en juillet ?",
  "answer": "Le Festival Jazz en Provence se tient à Aix-en-Provence du 15 au 20 juillet...",
  "sources": [
    { "title": "Festival Jazz en Provence", "city": "Aix-en-Provence", "date": "2026-07-15" }
  ]
}
```

### Chaîne RAG (`api/rag.py`)

- **Retriever** : FAISS avec `k=4` documents les plus proches
- **Chain type** : `"stuff"` — les 4 chunks sont concaténés dans le contexte
- **LLM** : `ChatMistralAI(model="mistral-large-latest", temperature=0.3)`
- **Prompt système** : répond toujours en français, se présente comme assistant culturel
- **Import critique** : `from langchain_classic.chains import RetrievalQA` (pas `langchain.chains`)

---

## Interface utilisateur

Lance `uv run --group ui python ui/app.py` (nécessite l'API sur `:8000`).

Interface Gradio avec 3 onglets :
- **Chat** : chatbot avec historique, suggestions d'exemples, tableau des sources
- **Santé de l'API** : vérification de l'état de l'index
- **Reconstruire l'index** : déclenchement du pipeline via `/rebuild`

URL de l'API configurable via `API_BASE_URL` (défaut : `http://localhost:8000`).

---

## Configuration

### Variables d'environnement (`.env`)

| Variable | Obligatoire | Défaut | Usage |
|----------|------------|--------|-------|
| `MISTRAL_API_KEY` | Oui | — | Embeddings + LLM (validée au démarrage) |
| `RUN_MODE` | Non | `production` | Sélectionne `config.toml` ou `debug.toml` |
| `DEBUG` | Non | `false` | Logs JSON (prod) vs texte (debug) |

### Fichiers de config (`config.toml` / `debug.toml`)

| Paramètre | Production | Debug |
|-----------|-----------|-------|
| `page_size` | 100 | 10 |
| `chunk_size` | 512 | 256 |
| `chunk_overlap` | 50 | 30 |
| `index_dir` | `vector_store/` | `vector_store_debug/` |

### Mode debug

```bash
# .env
RUN_MODE=debug
DEBUG=true
```

Utilise `tests/fixtures/debug_events_clean.csv` (2 événements) et `vector_store_debug/`.

---

## Installation et démarrage

### Prérequis

- Python >= 3.10
- `uv` (gestionnaire de paquets)
- Clé API Mistral dans `.env`

### Installation

```bash
uv sync
```

### Pipeline complet

```bash
uv run python scripts/fetch_events.py    # fetch → data/events_raw.json
uv run python scripts/clean_events.py   # clean → data/events_clean.csv
uv run python scripts/build_index.py    # vectorise → vector_store/
```

### Démarrage de l'API

```bash
uv run uvicorn api.main:app --reload
```

### Interface graphique

```bash
uv run --group ui python ui/app.py
```

---

## Tests

```bash
uv run pytest                      # tests unitaires (hors intégration)
uv uv run python -m tests.integration.evaluate_rag   # évaluation Ragas (nécessite MISTRAL_API_KEY + index)
uv run pre-commit run --all-files  # lint + typage + tests
```

### Structure des tests

| Fichier | Portée | Stratégie |
|---------|--------|-----------|
| `test_fetch.py` | 11 tests | Mocks `requests.get`, vérifie filtres et pagination |
| `test_clean.py` | 11 tests | Mocks `PATH`, teste nettoyage HTML et validation |
| `test_index.py` | 10 tests | Mix réel/mocké, `FakeEmbeddings` pour FAISS |
| `test_api.py` | 6 tests | Validation des schémas Pydantic uniquement |

Le mode debug (`RUN_MODE=debug`) est activé par défaut dans les tests via `tests/fixtures/debug_events_clean.csv` et les titres codés en dur ("Festival Jazz en Provence").

### Évaluation RAG (`tests/integration/evaluate_rag.py`)

Utilise [Ragas](https://docs.ragas.io) sur 12 paires Q/R annotées (`docs/test_set.json`) :

| Métrique | Description |
|----------|-------------|
| `Faithfulness` | La réponse est-elle fidèle aux documents récupérés ? |
| `AnswerRelevancy` | La réponse répond-elle bien à la question ? |
| `ContextRecall` | Les documents pertinents ont-ils été retrouvés ? |

Résultats exportés dans `docs/evaluation_results.csv`.

---

## Docker

### Prérequis

L'index vectoriel doit être construit **avant** de lancer le conteneur (voir [Pipeline](#pipeline)). Il sera monté en volume.

### Méthode recommandée : docker compose

```bash
# 1. Construire l'index vectoriel en local (une seule fois)
uv run python scripts/fetch_events.py
uv run python scripts/clean_events.py
uv run python scripts/build_index.py

# 2. Premier lancement : build des images + démarrage
sudo docker compose up --build

# 3. Lancements suivants : réutilise les images existantes
sudo docker compose up           # logs dans le terminal (Ctrl+C pour arrêter)
sudo docker compose up -d        # mode détaché (rend la main immédiatement)

# Suivi des logs en mode détaché
sudo docker compose logs -f      # tous les services
sudo docker compose logs api     # service spécifique

# Arrêt
sudo docker compose down
```

`docker compose` démarre deux services :

| Service | URL | Description |
|---------|-----|-------------|
| `api` | http://localhost:8000 | API FastAPI (Swagger sur `/docs`) |
| `ui` | http://localhost:7860 | Interface Gradio |

Le service `ui` communique avec `api` via le réseau Docker interne (`http://api:8000`). Le fichier `.env` est lu automatiquement pour injecter `MISTRAL_API_KEY` dans le service `api`. Les répertoires `vector_store/` et `data/` sont montés en volumes.

### Alternative : docker run

```bash
# Build de l'image API
docker build -t agenda-culturel-13 .

# Run API (monte le vector_store pré-construit)
docker run -p 8000:8000 \
  -e MISTRAL_API_KEY=sk-... \
  -v $(pwd)/vector_store:/app/vector_store \
  -v $(pwd)/data:/app/data \
  agenda-culturel-13

# Build et run de l'UI (dans un second terminal)
docker build -t agenda-culturel-13-ui -f ui/Dockerfile .
docker run -p 7860:7860 \
  -e API_BASE_URL=http://host.docker.internal:8000 \
  agenda-culturel-13-ui
```

L'image API embarque `api/`, `scripts/`, `logging_config.py`, `config.py` et `config.toml`. L'index vectoriel est monté en volume pour éviter de le reconstruire à chaque déploiement.

---

## Choix architecturaux et alternatives envisageables

### Source de données : OpenDataSoft / OpenAgenda

**Choix retenu** : API publique OpenDataSoft sur le jeu de données `evenements-publics-openagenda`. Aucune clé API requise, accès gratuit et illimité en lecture.

**Alternatives**
| Alternative | Avantages | Inconvénients |
|-------------|-----------|---------------|
| API OpenAgenda native | Données plus riches, webhooks disponibles | Clé API requise, quota |
| Scraping de sites locaux | Couverture maximale | Fragile, juridiquement risqué |
| Base de données locale propre | Contrôle total | Coût de maintenance élevé |

---

### Embeddings : `mistral-embed`

**Choix retenu** : `MistralAIEmbeddings(model="mistral-embed")` — vecteurs de dimension 1024, cohérent avec le LLM Mistral déjà utilisé (un seul fournisseur, une seule clé API).

**Alternatives**
| Alternative | Dimension | Avantages | Inconvénients |
|-------------|-----------|-----------|---------------|
| `text-embedding-3-small` (OpenAI) | 1536 | Très performant, bien documenté | Dépendance OpenAI, coût |
| `all-MiniLM-L6-v2` (Sentence-Transformers) | 384 | Gratuit, local, rapide | Moins performant sur le français |
| `camembert-base` | 768 | Optimisé français | Modèle lourd, hébergement requis |
| `multilingual-e5-large` | 1024 | Excellent multilingue | Ressources GPU recommandées |

Pour un usage offline ou sur données sensibles, `all-MiniLM-L6-v2` ou `paraphrase-multilingual-mpnet-base-v2` seraient de bons choix locaux.

---

### Base vectorielle : FAISS

**Choix retenu** : FAISS (Facebook AI Similarity Search) — bibliothèque locale, zéro infrastructure, index sauvegardé sur disque. Suffisant pour un corpus de quelques milliers d'événements.

FAISS est le choix le plus simple pour un prototype local. En production avec des besoins de filtrage sur les métadonnées (par ville, par date), **Qdrant** ou **Chroma** offriraient une meilleure ergonomie.

---

### Stratégie de récupération (Retrieval)

**Choix retenu** : recherche par similarité cosinus pure avec `k=4` (dense retrieval, `"stuff"` chain type).

**Alternatives et améliorations possibles**

| Stratégie | Description | Gain attendu |
|-----------|-------------|--------------|
| **Hybrid search (BM25 + vecteurs)** | Combine score lexical et sémantique | Meilleur rappel sur noms propres (villes, artistes) |
| **Reranking** | Cross-encoder après retrieval (ex. `cross-encoder/ms-marco`) | Meilleure précision sur les top-k |
| **Filtrage par métadonnées** | Pré-filtre FAISS sur `city` ou `firstdate_begin` avant la recherche vectorielle | Pertinence géographique/temporelle garantie |
| **MMR (Maximal Marginal Relevance)** | Diversifie les documents récupérés | Évite les doublons de chunks du même événement |
| **HyDE (Hypothetical Document Embeddings)** | Génère un document hypothétique avant de chercher | Améliore la recherche sur des questions vagues |
| **k adaptatif** | Augmenter k (8-10) avec reranking aval | Meilleur rappel sans surcharger le contexte |

---

### LLM : `mistral-large-latest`

**Choix retenu** : `ChatMistralAI(model="mistral-large-latest", temperature=0.3)` — modèle performant en français, même fournisseur que les embeddings.

**Alternatives**

| Alternative | Avantages | Inconvénients |
|-------------|-----------|---------------|
| `mistral-small` | Moins cher, plus rapide | Moins précis sur les nuances |
| `gpt-4o` (OpenAI) | Excellent, multimodal | Dépendance OpenAI, coût plus élevé |
| `llama-3.1-70b` (via Ollama) | Gratuit, local, vie privée | Infrastructure GPU requise |
| `gemma-2-9b` (local) | Léger, open-source | Qualité moindre en français |

---

### Chain type : `"stuff"`

**Choix retenu** : tous les documents récupérés (k=4 chunks) sont concaténés dans un seul contexte avant d'être envoyés au LLM.

**Alternatives**

| Type | Description | Quand l'utiliser |
|------|-------------|-----------------|
| **`stuff`** (retenu) | Tout dans un prompt | Petit contexte, k faible |
| **`map_reduce`** | LLM sur chaque doc, puis synthèse | Grand nombre de documents |
| **`refine`** | Raffine la réponse doc par doc | Réponses longues et détaillées |
| **Conversational RAG** | Historique de conversation inclus | Chatbot multi-tour |

Avec `k=4` chunks courts (512 tokens), `"stuff"` est adapté. Si `k` augmentait à 10+, `map_reduce` ou un modèle à grande fenêtre contextuelle serait préférable.

---

### Découpage des documents (Chunking)

**Choix retenu** : `RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)` sur `"{title}. {description}"`.

**Alternatives**

| Stratégie | Description | Avantages |
|-----------|-------------|-----------|
| **Document entier** (pas de chunking) | 1 doc = 1 événement | Pas de perte de contexte inter-chunks |
| **Chunking sémantique** | Découpe sur les changements de thème | Chunks plus cohérents sémantiquement |
| **Parent-child chunks** | Indexe de petits chunks, récupère le parent | Précision + contexte complet |
| **Chunking par champ** | Chunks séparés pour titre, description, lieu | Contrôle fin de la granularité |

Pour des événements culturels dont la description tient souvent en moins de 512 tokens, le chunking par document entier serait ici une alternative valide — et éviterait la perte de contexte entre chunks d'un même événement.

---

### Infrastructure de déploiement

**Choix retenu** : Docker + montage du `vector_store/` en volume. L'index est construit hors conteneur et partagé.

---

### Résumé des choix de simplification

Le projet privilégie la **simplicité opérationnelle** : un seul fournisseur (Mistral), un index local (FAISS), pas de base de données. Ce sont des choix raisonnables pour un prototype ou un usage à faible volume.

Les axes d'amélioration prioritaires pour une mise en production seraient :
1. **Filtrage par métadonnées** (ville, dates) dans le retriever pour améliorer la pertinence
2. **Hybrid search** (BM25 + vecteurs) pour les requêtes avec noms propres
3. **Remplacement de FAISS** par Qdrant ou Chroma pour le filtrage et la scalabilité
4. **Mise à jour incrémentale de l'index** (ne re-vectoriser que les nouveaux événements)
5. **Conversational RAG** pour conserver le contexte entre les questions
