"""Pure-Python retrieval logic — no FastAPI dependency.

Both the HTTP routes (`app/routes/*.py`) and the intelligence pipeline
(`app/intelligence/pipeline.py`) call into here. Returns plain dicts so it can
be invoked from a CLI without a request context.
"""
import uuid
from typing import Any, Optional

from app.db import get_pool
from app.embeddings import embed_query


async def hybrid_search(
    query: str,
    *,
    top_k: int = 5,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    section_types: Optional[list[str]] = None,
    domain: Optional[str] = None,
    field: Optional[str] = None,
    subfield: Optional[str] = None,
    paper_id: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    is_abstract: Optional[bool] = None,
) -> list[dict[str, Any]]:
    """Hybrid pgvector + tsvector search. Score is a weighted combination of
    cosine similarity (1 - distance) and PostgreSQL full-text rank.

    `section_types` filters via `section_type = ANY($n)` — pass a list to keep
    a query bounded to e.g. ['limitation', 'discussion', 'conclusion'].
    """
    query_vec = embed_query(query)
    params: list[Any] = [query_vec, semantic_weight, query, keyword_weight]
    conditions: list[str] = []
    idx = 5

    if section_types:
        conditions.append(f"section_type = ANY(${idx})")
        params.append(section_types)
        idx += 1

    for col, val in (
        ("domain", domain),
        ("field", field),
        ("subfield", subfield),
        ("paper_id", paper_id),
    ):
        if val is not None:
            conditions.append(f"{col} = ${idx}")
            params.append(val)
            idx += 1

    if year_min is not None:
        conditions.append(f"year >= ${idx}")
        params.append(year_min)
        idx += 1
    if year_max is not None:
        conditions.append(f"year <= ${idx}")
        params.append(year_max)
        idx += 1
    if is_abstract is not None:
        conditions.append(f"is_abstract = ${idx}")
        params.append(is_abstract)
        idx += 1

    where_clause = " AND ".join(conditions) if conditions else "TRUE"
    params.append(top_k)
    limit_idx = idx

    sql = f"""
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
            field,
            subfield,
            citations,
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
        rows = await conn.fetch(sql, *params)

    return [
        {
            "chunk_id": str(r["chunk_id"]),
            "position": r["position"],
            "text": r["text"],
            "section_title": r["section_title"],
            "section_type": r["section_type"],
            "is_abstract": r["is_abstract"],
            "paper_id": r["paper_id"],
            "title": r["title"],
            "year": r["year"],
            "authors": list(r["authors"]) if r["authors"] else [],
            "doi": r["doi"],
            "domain": r["domain"],
            "field": r["field"],
            "subfield": r["subfield"],
            "citations": r["citations"],
            "score": float(r["score"]),
        }
        for r in rows
    ]


async def get_context(chunk_id: str) -> Optional[dict[str, Any]]:
    """Return the target chunk plus its immediate neighbors (position ±1
    within the same paper). Returns None if the chunk doesn't exist.
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        target = await conn.fetchrow(
            "SELECT * FROM chunks WHERE id = $1",
            uuid.UUID(chunk_id),
        )
        if target is None:
            return None
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

    before = after = None
    for row in neighbors:
        if row["position"] == target["position"] - 1:
            before = _row_to_chunk(row)
        elif row["position"] == target["position"] + 1:
            after = _row_to_chunk(row)

    return {"target": _row_to_chunk(target), "before": before, "after": after}


async def get_paper_chunks(paper_id: str) -> list[dict[str, Any]]:
    """All chunks for a paper, ordered by position. Empty list if paper unknown."""
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM chunks WHERE paper_id = $1 ORDER BY position",
            paper_id,
        )
    return [_row_to_chunk(r) for r in rows]


def _row_to_chunk(row) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "position": row["position"],
        "text": row["text"],
        "section_title": row["section_title"],
        "section_type": row["section_type"],
        "section_index": row["section_index"],
        "chunk_index": row["chunk_index"],
        "is_abstract": row["is_abstract"],
        "paper_id": row["paper_id"],
        "title": row["title"],
        "year": row["year"],
        "authors": list(row["authors"]) if row["authors"] else [],
        "doi": row["doi"],
        "domain": row["domain"],
        "field": row["field"],
        "subfield": row["subfield"],
        "citations": row["citations"],
        "created_at": row["created_at"],
    }
