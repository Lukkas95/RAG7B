import uuid
from typing import List

from fastapi import APIRouter, HTTPException

from app.db import get_pool
from app.embeddings import embed_texts
from app.models import ChunkBatchIngest, ChunkResponse, ContextResponse

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
                    paper_id, title, year, authors, venue,
                    domain, field, subfield, topics, citations,
                    embedding
                ) VALUES (
                    $1, $2, $3, $4, $5,
                    $6, $7, $8, $9, $10,
                    $11, $12, $13, $14, $15,
                    $16
                )
                ON CONFLICT (id) DO UPDATE SET
                    position = EXCLUDED.position,
                    text = EXCLUDED.text,
                    section_title = EXCLUDED.section_title,
                    section_type = EXCLUDED.section_type,
                    paper_id = EXCLUDED.paper_id,
                    title = EXCLUDED.title,
                    year = EXCLUDED.year,
                    authors = EXCLUDED.authors,
                    venue = EXCLUDED.venue,
                    domain = EXCLUDED.domain,
                    field = EXCLUDED.field,
                    subfield = EXCLUDED.subfield,
                    topics = EXCLUDED.topics,
                    citations = EXCLUDED.citations,
                    embedding = EXCLUDED.embedding
                """,
                uuid.UUID(chunk_id),
                chunk.position,
                chunk.text,
                chunk.section_title,
                chunk.section_type,
                chunk.paper_id,
                chunk.title,
                chunk.year,
                chunk.authors,
                chunk.venue,
                chunk.domain,
                chunk.field,
                chunk.subfield,
                chunk.topics,
                chunk.citations,
                embeddings[i],
            )

    return {"inserted": len(batch.chunks)}


@router.get("/chunks/{chunk_id}/context", response_model=ContextResponse)
async def get_chunk_context(chunk_id: str):
    pool = get_pool()
    async with pool.acquire() as conn:
        target = await conn.fetchrow(
            "SELECT * FROM chunks WHERE id = $1",
            uuid.UUID(chunk_id),
        )
        if target is None:
            raise HTTPException(status_code=404, detail="Chunk not found")

        neighbors = await conn.fetch(
            """
            SELECT * FROM chunks
            WHERE paper_id = $1
              AND position BETWEEN $2 AND $3
            ORDER BY position
            """,
            target["paper_id"],
            target["position"] - 1,
            target["position"] + 1,
        )

    before = None
    after = None
    for row in neighbors:
        if row["position"] == target["position"] - 1:
            before = _row_to_chunk(row)
        elif row["position"] == target["position"] + 1:
            after = _row_to_chunk(row)

    return ContextResponse(
        target=_row_to_chunk(target),
        before=before,
        after=after,
    )


@router.get("/papers/{paper_id:path}/chunks", response_model=List[ChunkResponse])
async def get_paper_chunks(paper_id: str):
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM chunks WHERE paper_id = $1 ORDER BY position",
            paper_id,
        )
    if not rows:
        raise HTTPException(status_code=404, detail="No chunks found for this paper")
    return [_row_to_chunk(r) for r in rows]


def _row_to_chunk(row) -> dict:
    return {
        "id": str(row["id"]),
        "position": row["position"],
        "text": row["text"],
        "section_title": row["section_title"],
        "section_type": row["section_type"],
        "paper_id": row["paper_id"],
        "title": row["title"],
        "year": row["year"],
        "authors": row["authors"],
        "venue": row["venue"],
        "domain": row["domain"],
        "field": row["field"],
        "subfield": row["subfield"],
        "topics": row["topics"],
        "citations": row["citations"],
        "created_at": row["created_at"],
    }
