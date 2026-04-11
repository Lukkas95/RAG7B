-- Enable pgvector extension (must come before any vector column definitions)
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table: one row per ingested paper/article
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    authors TEXT,
    year INTEGER,
    source TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chunks table: text segments with embeddings and full-text search
-- NOTE: embedding dimension is 384 (all-MiniLM-L6-v2). To change model,
-- drop the HNSW index, ALTER COLUMN embedding TYPE vector(NEW_DIM), recreate index.
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    section_label VARCHAR(255),
    embedding vector(384),
    content_tsv TSVECTOR,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for fast approximate nearest neighbor search (cosine distance)
CREATE INDEX idx_chunks_embedding ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN index for full-text search
CREATE INDEX idx_chunks_tsv ON chunks USING gin (content_tsv);

-- Composite index for context reconstruction (fetch neighboring chunks)
CREATE INDEX idx_chunks_doc_idx ON chunks (document_id, chunk_index);

-- Index for filtered search by section
CREATE INDEX idx_chunks_section ON chunks (section_label);

-- Auto-generate tsvector from content on insert/update
CREATE OR REPLACE FUNCTION chunks_tsv_trigger_fn()
RETURNS TRIGGER AS $$
BEGIN
    NEW.content_tsv := to_tsvector('english', NEW.content);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER chunks_tsv_trigger
    BEFORE INSERT OR UPDATE ON chunks
    FOR EACH ROW
    EXECUTE FUNCTION chunks_tsv_trigger_fn();
