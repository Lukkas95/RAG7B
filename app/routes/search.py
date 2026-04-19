from fastapi import APIRouter

from app.db import get_pool
from app.embeddings import embed_query
from app.models import SearchRequest, SearchResponse, SearchResult

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def hybrid_search(req: SearchRequest):
    query_vec = embed_query(req.query)

    # Build dynamic WHERE clause with parameterized filters
    # First 4 params are always: query_vec, semantic_weight, query_text, keyword_weight
    params = [query_vec, req.semantic_weight, req.query, req.keyword_weight]
    conditions = []
    idx = 5  # next parameter index

    filters = {
        "section_type": req.section_type if req.section_type else None,
        "domain": req.domain if req.domain else None,
        "field": req.field if req.field else None,
        "subfield": req.subfield if req.subfield else None,
        "paper_id": req.paper_id if req.paper_id else None,
    }

    for col, val in filters.items():
        if val is not None:
            conditions.append(f"{col} = ${idx}")
            params.append(val)
            idx += 1

    if req.year_min is not None:
        conditions.append(f"year >= ${idx}")
        params.append(req.year_min)
        idx += 1

    if req.year_max is not None:
        conditions.append(f"year <= ${idx}")
        params.append(req.year_max)
        idx += 1

    if req.is_abstract is not None:
        conditions.append(f"is_abstract = ${idx}")
        params.append(req.is_abstract)
        idx += 1

    where_clause = " AND ".join(conditions) if conditions else "TRUE"

    params.append(req.top_k)
    limit_idx = idx

    query = f"""
        SELECT
            id AS chunk_id,
            position,
            text,
            section_title,
            section_type,
            is_abstract,
            paper_id,
            title,
            year,
            authors,
            doi,
            domain,
            (
                (1 - (embedding <=> $1::vector)) * $2
                + ts_rank(content_tsv, plainto_tsquery('english', $3)) * $4
            ) AS score
        FROM chunks
        WHERE {where_clause}
        ORDER BY score DESC
        LIMIT ${limit_idx}
    """

    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    results = [
        SearchResult(
            chunk_id=str(r["chunk_id"]),
            position=r["position"],
            text=r["text"],
            section_title=r["section_title"],
            section_type=r["section_type"],
            is_abstract=r["is_abstract"],
            paper_id=r["paper_id"],
            title=r["title"],
            year=r["year"],
            authors=r["authors"],
            doi=r["doi"],
            domain=r["domain"],
            score=float(r["score"]),
        )
        for r in rows
    ]
    return SearchResponse(query=req.query, results=results)
