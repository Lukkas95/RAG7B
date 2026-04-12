-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Single denormalized table: each row is one chunk with its paper metadata
-- NOTE: embedding dimension is 384 (all-MiniLM-L6-v2). To change model,
-- drop the HNSW index, ALTER COLUMN embedding TYPE vector(NEW_DIM), recreate index.
CREATE TABLE chunks (
    -- Chunk identity
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    position        INTEGER NOT NULL,
    text            TEXT NOT NULL,
    section_title   VARCHAR(255),
    section_type    VARCHAR(100),

    -- Paper-level metadata (denormalized)
    paper_id        TEXT NOT NULL,
    title           TEXT NOT NULL,
    year            INTEGER,
    authors         TEXT[] NOT NULL DEFAULT '{}',
    venue           TEXT,
    domain          TEXT,
    field           TEXT,
    subfield        TEXT,
    topics          TEXT[] NOT NULL DEFAULT '{}',
    citations       INTEGER DEFAULT 0,

    -- Search infrastructure
    embedding       vector(384),
    content_tsv     TSVECTOR,

    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for fast approximate nearest neighbor search (cosine distance)
CREATE INDEX idx_chunks_embedding ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN index for full-text search
CREATE INDEX idx_chunks_tsv ON chunks USING gin (content_tsv);

-- Context reconstruction: fetch neighboring chunks within same paper
CREATE INDEX idx_chunks_paper_position ON chunks (paper_id, position);

-- Metadata filter indexes
CREATE INDEX idx_chunks_section_type ON chunks (section_type);
CREATE INDEX idx_chunks_domain ON chunks (domain);
CREATE INDEX idx_chunks_field ON chunks (field);
CREATE INDEX idx_chunks_subfield ON chunks (subfield);
CREATE INDEX idx_chunks_year ON chunks (year);
CREATE INDEX idx_chunks_paper_id ON chunks (paper_id);

-- GIN index on topics array for containment queries (@>)
CREATE INDEX idx_chunks_topics ON chunks USING gin (topics);

-- Auto-generate tsvector from text on insert/update
CREATE OR REPLACE FUNCTION chunks_tsv_trigger_fn()
RETURNS TRIGGER AS $$
BEGIN
    NEW.content_tsv := to_tsvector('english', NEW.text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER chunks_tsv_trigger
    BEFORE INSERT OR UPDATE ON chunks
    FOR EACH ROW
    EXECUTE FUNCTION chunks_tsv_trigger_fn();
