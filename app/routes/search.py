from fastapi import APIRouter

from app.db import get_pool
from app.embeddings import embed_query
from app.models import SearchRequest, SearchResponse, SearchResult

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def hybrid_search(req: SearchRequest):
    query_vec = embed_query(req.query)

    pool = get_pool()
    async with pool.acquire() as conn:
        if req.section_label is not None:
            rows = await conn.fetch(
                """
                SELECT
                    c.id AS chunk_id,
                    c.document_id,
                    c.chunk_index,
                    c.content,
                    c.section_label,
                    d.title AS document_title,
                    (
                        (1 - (c.embedding <=> $1::vector)) * $2
                        + ts_rank(c.content_tsv, plainto_tsquery('english', $3)) * $4
                    ) AS score
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.section_label = $5
                ORDER BY score DESC
                LIMIT $6
                """,
                query_vec, req.semantic_weight, req.query,
                req.keyword_weight, req.section_label, req.top_k,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT
                    c.id AS chunk_id,
                    c.document_id,
                    c.chunk_index,
                    c.content,
                    c.section_label,
                    d.title AS document_title,
                    (
                        (1 - (c.embedding <=> $1::vector)) * $2
                        + ts_rank(c.content_tsv, plainto_tsquery('english', $3)) * $4
                    ) AS score
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                ORDER BY score DESC
                LIMIT $5
                """,
                query_vec, req.semantic_weight, req.query,
                req.keyword_weight, req.top_k,
            )

    results = [
        SearchResult(
            chunk_id=r["chunk_id"],
            document_id=r["document_id"],
            chunk_index=r["chunk_index"],
            content=r["content"],
            section_label=r["section_label"],
            score=float(r["score"]),
            document_title=r["document_title"],
        )
        for r in rows
    ]
    return SearchResponse(query=req.query, results=results)
