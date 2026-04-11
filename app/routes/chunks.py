from fastapi import APIRouter, HTTPException

from app.db import get_pool
from app.embeddings import embed_texts
from app.models import ChunkBatchCreate, ChunkResponse, ContextResponse

router = APIRouter(tags=["chunks"])


@router.post("/documents/{document_id}/chunks", status_code=201)
async def ingest_chunks(document_id: int, batch: ChunkBatchCreate):
    pool = get_pool()
    async with pool.acquire() as conn:
        # Verify document exists
        doc = await conn.fetchval(
            "SELECT id FROM documents WHERE id = $1", document_id
        )
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")

        # Batch-compute embeddings for all chunks at once
        contents = [c.content for c in batch.chunks]
        embeddings = embed_texts(contents)

        # Insert all chunks
        await conn.executemany(
            """
            INSERT INTO chunks (document_id, chunk_index, content, section_label, embedding)
            VALUES ($1, $2, $3, $4, $5)
            """,
            [
                (
                    document_id,
                    chunk.chunk_index,
                    chunk.content,
                    chunk.section_label,
                    embeddings[i],
                )
                for i, chunk in enumerate(batch.chunks)
            ],
        )

    return {"inserted": len(batch.chunks)}


@router.get("/chunks/{chunk_id}/context", response_model=ContextResponse)
async def get_chunk_context(chunk_id: int):
    pool = get_pool()
    async with pool.acquire() as conn:
        # Fetch the target chunk
        target = await conn.fetchrow(
            "SELECT * FROM chunks WHERE id = $1", chunk_id
        )
        if target is None:
            raise HTTPException(status_code=404, detail="Chunk not found")

        # Fetch neighbors by chunk_index within the same document
        neighbors = await conn.fetch(
            """
            SELECT * FROM chunks
            WHERE document_id = $1
              AND chunk_index BETWEEN $2 AND $3
            ORDER BY chunk_index
            """,
            target["document_id"],
            target["chunk_index"] - 1,
            target["chunk_index"] + 1,
        )

    # Separate into before / target / after
    before = None
    after = None
    target_dict = dict(target)
    for row in neighbors:
        if row["chunk_index"] == target["chunk_index"] - 1:
            before = dict(row)
        elif row["chunk_index"] == target["chunk_index"] + 1:
            after = dict(row)

    return ContextResponse(
        target=target_dict,
        before=before,
        after=after,
    )
