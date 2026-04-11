# ScholarGraph RAG — Database & Retrieval Service

A pgvector-powered retrieval backend for the ScholarGraph RAG system. Provides hybrid search (semantic + keyword) over academic literature via a FastAPI REST API.

Built for the RAG course project — Role B (Database & Retrieval Architect).

## Features

- **Hybrid Search**: Combines pgvector cosine similarity with PostgreSQL full-text search (`tsvector`/`ts_rank`), with configurable weights
- **Section Filtering**: Search within specific paper sections (e.g., "Methodology", "Results")
- **Context Reconstruction**: Retrieve a chunk along with its neighboring chunks for full argument context
- **Batch Ingestion**: Upload document chunks with automatic embedding computation (server-side)
- **Interactive API Docs**: Auto-generated Swagger UI at `/docs`

## Tech Stack

- **PostgreSQL 16** with **pgvector** (Docker)
- **FastAPI** + **asyncpg**
- **sentence-transformers** (`all-MiniLM-L6-v2`, 384-dim embeddings)

## Quickstart

### Prerequisites

- Docker & Docker Compose
- Python 3.9+

### 1. Start the database

```bash
docker compose up -d
```

This launches PostgreSQL with pgvector and creates the schema automatically.

### 2. Install Python dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Start the API server

```bash
uvicorn app.main:app --reload --port 8000
```

The first startup downloads the embedding model (~80MB). Once ready, visit:
- **API**: http://localhost:8000
- **Interactive docs**: http://localhost:8000/docs

### 4. Load demo data (optional)

```bash
python3 scripts/load_demo_data.py
```

Loads 3,200 Wikipedia passages from the `rag-datasets/rag-mini-wikipedia` HuggingFace dataset.

## API Reference

### Documents

#### `POST /documents`
Register a new document.

```json
{
  "title": "My Paper",
  "authors": "Alice, Bob",
  "year": 2024,
  "source": "arxiv"
}
```

#### `GET /documents`
List all documents.

---

### Chunks

#### `POST /documents/{document_id}/chunks`
Ingest text chunks for a document. Embeddings are computed server-side.

```json
{
  "chunks": [
    {"chunk_index": 0, "content": "Introduction text...", "section_label": "Introduction"},
    {"chunk_index": 1, "content": "Methods text...", "section_label": "Methods"}
  ]
}
```

- Max 500 chunks per request
- `section_label` is optional but enables filtered search

#### `GET /chunks/{chunk_id}/context`
**Context Reconstructor** — returns the target chunk plus its immediate neighbors (chunk_index ± 1).

```json
{
  "target": {"id": 2, "content": "...", "section_label": "Methods", ...},
  "before": {"id": 1, "content": "...", "section_label": "Introduction", ...},
  "after":  {"id": 3, "content": "...", "section_label": "Results", ...}
}
```

---

### Search

#### `POST /search`
Hybrid search combining semantic similarity with keyword matching.

```json
{
  "query": "vector database performance",
  "top_k": 5,
  "semantic_weight": 0.7,
  "keyword_weight": 0.3,
  "section_label": "Methods"
}
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `query` | required | Search query text |
| `top_k` | 5 | Number of results (1-50) |
| `semantic_weight` | 0.7 | Weight for vector similarity |
| `keyword_weight` | 0.3 | Weight for keyword matching |
| `section_label` | null | Filter by section (optional) |

**Response:**
```json
{
  "query": "vector database performance",
  "results": [
    {
      "chunk_id": 42,
      "document_id": 3,
      "chunk_index": 5,
      "content": "pgvector adds vector similarity search...",
      "section_label": "Methods",
      "score": 0.47,
      "document_title": "My Paper"
    }
  ]
}
```

---

### Health

#### `GET /health`
Returns `{"status": "ok"}` if the database connection is alive.

## Database Schema

```
documents: id, title, authors, year, source, created_at
chunks:    id, document_id, chunk_index, content, section_label, embedding, content_tsv, created_at
```

**Indexes:**
- HNSW on `embedding` (cosine distance) — fast approximate nearest neighbor
- GIN on `content_tsv` — full-text search
- B-tree on `(document_id, chunk_index)` — context reconstruction
- B-tree on `section_label` — filtered search

## Project Structure

```
├── docker-compose.yml          # pgvector PostgreSQL
├── init.sql                    # schema, indexes, triggers
├── requirements.txt
├── app/
│   ├── main.py                 # FastAPI app + lifespan
│   ├── db.py                   # asyncpg connection pool
│   ├── embeddings.py           # sentence-transformer wrapper
│   ├── models.py               # Pydantic request/response models
│   └── routes/
│       ├── documents.py        # document CRUD
│       ├── chunks.py           # chunk ingestion + context reconstructor
│       └── search.py           # hybrid search
└── scripts/
    └── load_demo_data.py       # demo data loader
```

## For Teammates

- **Ahreum (Ingestion)**: Use `POST /documents` + `POST /documents/{id}/chunks` to push parsed papers
- **Yerim (Intelligence)**: Use `POST /search` with section filters + `GET /chunks/{id}/context` for retrieval
- **Matthias (UI)**: Use all endpoints; interactive Swagger docs at `/docs`
