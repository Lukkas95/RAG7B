# ScholarGraph RAG — Database & Retrieval Service

A pgvector-powered retrieval backend for the ScholarGraph RAG system. Provides hybrid search (semantic + keyword) over academic literature via a FastAPI REST API.

Built for the RAG course project — Role B (Database & Retrieval Architect).

## Features

- **Hybrid Search**: Combines pgvector cosine similarity with PostgreSQL full-text search (`tsvector`/`ts_rank`), with configurable weights
- **Rich Metadata Filtering**: Filter by section type, domain, field, subfield, year range, paper ID, or abstract-only
- **Context Reconstruction**: Retrieve a chunk along with its neighboring chunks for full argument context
- **Batch Ingestion**: Upload chunks with automatic embedding computation (server-side)
- **Upsert Support**: Re-ingesting the same `chunk_id` updates instead of duplicating
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

### 4. Load data

```bash
python3 scripts/load_jsonl.py chunked_results.jsonl
```

Loads the JSONL file produced by Role A's pipeline. Data persists in the Docker volume — you only need to run this once.

For the demo dataset (optional):
```bash
python3 scripts/load_demo_data.py
```

## API Reference

### Chunks

#### `POST /chunks`
Ingest chunks with paper metadata. Embeddings are computed server-side. Supports upsert (re-ingesting the same `chunk_id` updates the existing row).

```json
{
  "chunks": [
    {
      "chunk_id": "f18cc432-38e0-448d-acd4-212dd7142410",
      "text": "Recent studies on vision Transformer...",
      "section_title": "Introduction",
      "section_type": "introduction",
      "section_index": 0,
      "chunk_index": 0,
      "position": 0,
      "is_abstract": false,
      "paper_id": "https://openalex.org/W3175515048",
      "title": "PVT v2: Improved baselines with pyramid vision transformer",
      "year": 2022,
      "authors": ["Wenhai Wang", "Enze Xie"],
      "doi": "https://doi.org/10.1007/s41095-022-0274-8",
      "domain": "Physical Sciences",
      "field": "Computer Science",
      "subfield": "Computer Vision and Pattern Recognition",
      "citations": 2060
    }
  ]
}
```

- Max 500 chunks per request
- `chunk_id` is optional (UUID generated server-side if omitted)

#### `GET /chunks/{chunk_id}/context`
**Context Reconstructor** — returns the target chunk plus its immediate neighbors (position +/- 1 within the same paper).

```json
{
  "target": {"id": "f18c...", "text": "...", "section_type": "introduction", ...},
  "before": null,
  "after":  {"id": "a1b2...", "text": "...", "section_type": "methods", ...}
}
```

#### `GET /papers/{paper_id}/chunks`
List all chunks for a given paper, ordered by position.

---

### Search

#### `POST /search`
Hybrid search combining semantic similarity with keyword matching.

```json
{
  "query": "vision transformer attention",
  "top_k": 5,
  "semantic_weight": 0.7,
  "keyword_weight": 0.3,
  "section_type": "method",
  "field": "Computer Science"
}
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `query` | required | Search query text |
| `top_k` | 5 | Number of results (1-50) |
| `semantic_weight` | 0.7 | Weight for vector similarity |
| `keyword_weight` | 0.3 | Weight for keyword matching |
| `section_type` | null | Filter by section type (e.g., "introduction", "method", "result") |
| `domain` | null | Filter by domain (e.g., "Physical Sciences") |
| `field` | null | Filter by field (e.g., "Computer Science") |
| `subfield` | null | Filter by subfield |
| `year_min` | null | Filter by minimum year |
| `year_max` | null | Filter by maximum year |
| `paper_id` | null | Filter by specific paper |
| `is_abstract` | null | Filter abstract-only chunks (true/false) |

**Response:**
```json
{
  "query": "vision transformer attention",
  "results": [
    {
      "chunk_id": "f18cc432-38e0-448d-acd4-212dd7142410",
      "position": 0,
      "text": "Recent studies on vision Transformer...",
      "section_title": "Introduction",
      "section_type": "introduction",
      "is_abstract": false,
      "paper_id": "https://openalex.org/W3175515048",
      "title": "PVT v2: Improved baselines with pyramid vision transformer",
      "year": 2022,
      "authors": ["Wenhai Wang", "Enze Xie"],
      "doi": "https://doi.org/10.1007/s41095-022-0274-8",
      "domain": "Physical Sciences",
      "score": 0.465
    }
  ]
}
```

---

### Health

#### `GET /health`
Returns `{"status": "ok"}` if the database connection is alive.

## Database Schema

Single denormalized `chunks` table — each row is one chunk with its paper metadata:

```
chunks: id (UUID), position, text, section_title, section_type,
        section_index, chunk_index, is_abstract,
        paper_id, title, year, authors[], doi, domain, field,
        subfield, citations, embedding, content_tsv, created_at
```

**Indexes:**
- HNSW on `embedding` (cosine distance) — fast approximate nearest neighbor
- GIN on `content_tsv` — full-text search
- B-tree on `(paper_id, position)` — context reconstruction
- B-tree on `section_type`, `domain`, `field`, `subfield`, `year`, `paper_id`, `is_abstract` — filtered search

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
│       ├── chunks.py           # chunk ingestion + context reconstructor
│       └── search.py           # hybrid search
└── scripts/
    ├── load_jsonl.py           # load JSONL from Role A's pipeline
    └── load_demo_data.py       # demo data loader (Wikipedia)
```

## For Teammates

- **Ahreum (Ingestion)**: Run `python3 scripts/load_jsonl.py chunked_results.jsonl` to load your data, or use `POST /chunks` directly
- **Yerim (Intelligence)**: Use `POST /search` with filters + `GET /chunks/{id}/context` for retrieval
- **Matthias (UI)**: Use all endpoints; interactive Swagger docs at `/docs`
