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
- Git LFS (for large demo data, optional): https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage

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
python3 scripts/load_jsonl.py chunked_results_v2.jsonl
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
      "pdf_url": "https://arxiv.org/pdf/2106.13797",
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
      "pdf_url": "https://arxiv.org/pdf/2106.13797",
      "domain": "Physical Sciences",
      "score": 0.465
    }
  ]
}
```

---

### Analyze (end-to-end pipeline)

#### `POST /analyze`

Runs the full intelligence pipeline (query expansion → fan-out hybrid search → group-by-paper → gap synthesis) in one call. Slow (depends on the LLM backend; 30–90s with local Ollama+Qwen, faster with Gemini).

```json
{
  "query": "vision transformer attention efficiency",
  "top_k_per_query": 8
}
```

**Response:**

```json
{
  "query": "vision transformer attention efficiency",
  "expanded_queries": ["...", "...", "..."],
  "papers": [
    {
      "paper_id": "https://openalex.org/W...",
      "title": "...",
      "year": 2022,
      "authors": ["..."],
      "doi": "...",
      "pdf_url": "https://arxiv.org/pdf/...",
      "domain": "Physical Sciences",
      "field": "Computer Science",
      "subfield": "...",
      "citations": 2060,
      "chunks": [
        {
          "chunk_id": "...",
          "section_title": "Discussion",
          "section_type": "discussion",
          "position": 12,
          "text": "...",
          "score": 0.47
        }
      ]
    }
  ],
  "analysis": "### Per-Paper Limitations & Future Work\n..."
}
```

LLM backend / model are controlled by env vars on the server (see [Configuration](#configuration)) — the frontend doesn't pick them.

### Health

#### `GET /health`

Returns `{"status": "ok"}` if the database connection is alive.

### CORS

Allowed origins default to `http://localhost:3000`, `http://localhost:5173`, `http://localhost:8080` (covers Next.js, Vite, and common dev servers). Override via env var:

```bash
export CORS_ORIGINS="http://localhost:3000,https://my-frontend.example.com"
```

Set `CORS_ORIGINS="*"` to allow any origin (dev only).

## Intelligence Pipeline

Beyond the HTTP API, the package ships an end-to-end RAG reasoning pipeline that queries the database directly (no HTTP hop) and runs LLM-powered query expansion + cross-paper gap synthesis. Originally Role C's responsibility (`ScholarGraph-RAG`); merged in here so the integration is one Python import instead of an HTTP client.

```bash
# Gemini (default) — set your key in .env or export it
export GOOGLE_API_KEY=...
python3 scripts/run_pipeline.py "vision transformer attention efficiency"
```

The CLI writes a markdown report and a JSON dump under `output/`.

### Switching to a local LLM (Ollama)

For an open-source local model on Apple Silicon:

```bash
brew install ollama

# Start the daemon — pick one:
brew services start ollama                # background, restarts at login
# or, in a dedicated terminal:
ollama serve                              # foreground on :11434

# Then, with the daemon running:
ollama pull qwen2.5:7b-instruct          # ~5 GB, one-time
export LLM_BACKEND=ollama
python3 scripts/run_pipeline.py "vision transformer attention efficiency"
```

`qwen2.5:7b-instruct` was chosen for its strong JSON / structured-output behavior — important because the query-expansion step parses a JSON list out of the LLM's response. Llama 3.1 8B Instruct (`llama3.1:8b-instruct`) and Mistral 7B Instruct (`mistral:7b-instruct`) also work.

### Configuration

| Env var          | Default                                            | Notes                                             |
| ---------------- | -------------------------------------------------- | ------------------------------------------------- |
| `LLM_BACKEND`    | `gemini`                                           | `gemini` or `ollama`                              |
| `LLM_MODEL`      | backend-default                                    | e.g. `gemini-flash-latest`, `qwen2.5:7b-instruct` |
| `GOOGLE_API_KEY` | —                                                  | Required when `LLM_BACKEND=gemini`                |
| `OLLAMA_HOST`    | `http://localhost:11434`                           | Used when `LLM_BACKEND=ollama`                    |
| `DATABASE_URL`   | `postgresql://rag:rag@localhost:5432/scholargraph` |                                                   |

See `.env.example`.

### Pipeline stages

1. **Query expansion** (`app/intelligence/workflows.py:expand_query`) — one user query → 3 technical variations (LLM call).
2. **Retrieval** (`app/retrieval.py:hybrid_search`) — fan out the variants concurrently, biased to analytical sections (`section_type IN ('limitation','discussion','conclusion')`). Falls back to an unfiltered pass if fewer than 3 distinct papers come back.
3. **Group-by-paper** — chunks are deduped, bucketed under their `paper_id`, and joined with paper metadata.
4. **Gap synthesis** (`app/intelligence/workflows.py:gap_synthesis`) — single LLM call with strict citation rules; output cites every claim as `[Title (Year), §section_title]`.

Calling from Python directly (no CLI):

```python
import asyncio
from app.db import init_pool, close_pool
from app.intelligence.pipeline import run_pipeline

async def main():
    await init_pool()
    try:
        result = await run_pipeline("graph neural networks for citation prediction")
        print(result["analysis"])
    finally:
        await close_pool()

asyncio.run(main())
```

## Database Schema

Single denormalized `chunks` table — each row is one chunk with its paper metadata:

```
chunks: id (UUID), position, text, section_title, section_type,
        section_index, chunk_index, is_abstract,
        paper_id, title, year, authors[], doi, pdf_url, domain, field,
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
├── .env.example                # configuration reference
├── app/
│   ├── main.py                 # FastAPI app + lifespan
│   ├── db.py                   # asyncpg connection pool
│   ├── embeddings.py           # sentence-transformer wrapper
│   ├── models.py               # Pydantic request/response models
│   ├── retrieval.py            # pure-Python search + context (no FastAPI)
│   ├── routes/
│   │   ├── chunks.py           # ingest + thin wrappers around retrieval.py
│   │   └── search.py           # thin wrapper around retrieval.hybrid_search
│   └── intelligence/
│       ├── prompts.py          # expansion + gap-synthesis templates
│       ├── llm.py              # pluggable backend (Gemini / Ollama)
│       ├── workflows.py        # expand_query, gap_synthesis
│       └── pipeline.py         # end-to-end orchestrator
└── scripts/
    ├── load_jsonl.py           # load JSONL from Role A's pipeline
    ├── load_demo_data.py       # demo data loader (Wikipedia)
    └── run_pipeline.py         # CLI for the intelligence pipeline
```

## For Teammates

- **Ahreum (Ingestion)**: Run `python3 scripts/load_jsonl.py chunked_results.jsonl` to load your data, or use `POST /chunks` directly
- **Yerim (Intelligence)**: Your `expand_query_workflow` + `gap_synthesis_workflow` are now ported into `app/intelligence/`. Either run `scripts/run_pipeline.py "..."` or `from app.intelligence.pipeline import run_pipeline` — no HTTP needed.
- **Matthias (UI)**: One endpoint covers the full flow — `POST /analyze` with `{"query": "..."}` returns expanded queries, retrieved papers (with chunks), and the gap-synthesis analysis. Use `POST /search` if you want raw retrieval without the LLM step. Swagger UI at `/docs`. CORS pre-configured for ports 3000 / 5173 / 8080; override with `CORS_ORIGINS` env var.
