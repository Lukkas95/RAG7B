import uuid
from typing import List

from fastapi import APIRouter, HTTPException

from app.db import get_pool
from app.embeddings import embed_texts
from app.models import ChunkBatchIngest, ChunkResponse, ContextResponse
from app.retrieval import get_context, get_paper_chunks

router = APIRouter(tags=["chunks"])


@router.post("/chunks", status_code=201)
async def ingest_chunks(batch: ChunkBatchIngest):
    texts = [c.text for c in batch.chunks]
    embeddings = embed_texts(texts)

    pool = get_pool()
    async with pool.acquire() as conn:
        for i, chunk in enumerate(batch.chunks):
            chunk_id = chunk.chunk_id or str(uuid.uuid4())
            await conn.execute(
                """
                INSERT INTO chunks (
                    id, position, text, section_title, section_type,
                    section_index, chunk_index, is_abstract,
                    paper_id, title, year, authors, doi, pdf_url,
                    domain, field, subfield, citations,
                    embedding
                ) VALUES (
                    $1, $2, $3, $4, $5,
                    $6, $7, $8,
                    $9, $10, $11, $12, $13, $14,
                    $15, $16, $17, $18,
                    $19
                )
                ON CONFLICT (id) DO UPDATE SET
                    position = EXCLUDED.position,
                    text = EXCLUDED.text,
                    section_title = EXCLUDED.section_title,
                    section_type = EXCLUDED.section_type,
                    section_index = EXCLUDED.section_index,
                    chunk_index = EXCLUDED.chunk_index,
                    is_abstract = EXCLUDED.is_abstract,
                    paper_id = EXCLUDED.paper_id,
                    title = EXCLUDED.title,
                    year = EXCLUDED.year,
                    authors = EXCLUDED.authors,
                    doi = EXCLUDED.doi,
                    pdf_url = EXCLUDED.pdf_url,
                    domain = EXCLUDED.domain,
                    field = EXCLUDED.field,
                    subfield = EXCLUDED.subfield,
                    citations = EXCLUDED.citations,
                    embedding = EXCLUDED.embedding
                """,
                uuid.UUID(chunk_id),
                chunk.position,
                chunk.text,
                chunk.section_title,
                chunk.section_type,
                chunk.section_index,
                chunk.chunk_index,
                chunk.is_abstract,
                chunk.paper_id,
                chunk.title,
                chunk.year,
                chunk.authors,
                chunk.doi,
                chunk.pdf_url,
                chunk.domain,
                chunk.field,
                chunk.subfield,
                chunk.citations,
                embeddings[i],
            )

    return {"inserted": len(batch.chunks)}


@router.get("/chunks/{chunk_id}/context", response_model=ContextResponse)
async def chunk_context(chunk_id: str):
    ctx = await get_context(chunk_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return ContextResponse(**ctx)


@router.get("/papers/{paper_id:path}/chunks", response_model=List[ChunkResponse])
async def paper_chunks(paper_id: str):
    rows = await get_paper_chunks(paper_id)
    if not rows:
        raise HTTPException(status_code=404, detail="No chunks found for this paper")
    return [ChunkResponse(**r) for r in rows]
