"""End-to-end intelligence pipeline.

  user query
    -> expand_query()                                 (LLM)
    -> hybrid_search per variant, biased to analytical sections   (DB)
    -> fallback unfiltered pass if too few distinct papers found
    -> group chunks by paper, attach paper metadata
    -> gap_synthesis()                                (LLM)
"""
from typing import Any

import asyncio

from app.intelligence.llm import describe_backend
from app.intelligence.workflows import expand_query, gap_synthesis
from app.retrieval import hybrid_search

# Sections that tend to carry critical analysis / limitations / future work.
ANALYTICAL_SECTION_TYPES = ["limitation", "discussion", "conclusion"]


async def run_pipeline(
    user_query: str,
    *,
    top_k_per_query: int = 8,
    min_distinct_papers: int = 3,
    verbose: bool = True,
) -> dict[str, Any]:
    log = print if verbose else (lambda *a, **k: None)
    log(f"[pipeline] backend={describe_backend()} | query={user_query!r}")

    log("[pipeline] expanding query...")
    expanded = await expand_query(user_query)
    log(f"[pipeline] {len(expanded)} expanded queries: {expanded}")

    log(f"[pipeline] retrieving (filtered to {ANALYTICAL_SECTION_TYPES})...")
    chunk_lists = await asyncio.gather(
        *[
            hybrid_search(
                q, top_k=top_k_per_query, section_types=ANALYTICAL_SECTION_TYPES
            )
            for q in expanded
        ]
    )
    chunks = _dedupe_chunks([c for lst in chunk_lists for c in lst])
    distinct_papers = {c["paper_id"] for c in chunks}
    log(
        f"[pipeline] retrieved {len(chunks)} unique chunks "
        f"across {len(distinct_papers)} papers"
    )

    if len(distinct_papers) < min_distinct_papers:
        log(
            f"[pipeline] < {min_distinct_papers} papers — falling back to unfiltered search"
        )
        fallback_lists = await asyncio.gather(
            *[hybrid_search(q, top_k=top_k_per_query) for q in expanded]
        )
        chunks = _dedupe_chunks(chunks + [c for lst in fallback_lists for c in lst])
        log(
            f"[pipeline] after fallback: {len(chunks)} chunks across "
            f"{len({c['paper_id'] for c in chunks})} papers"
        )

    papers = _group_by_paper(chunks)
    log(f"[pipeline] grouped into {len(papers)} papers; calling synthesis LLM...")

    analysis = await gap_synthesis(papers)
    log("[pipeline] done.")

    return {
        "query": user_query,
        "expanded_queries": expanded,
        "papers": papers,
        "analysis": analysis,
    }


def _dedupe_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for c in chunks:
        if c["chunk_id"] in seen:
            continue
        seen.add(c["chunk_id"])
        out.append(c)
    return out


def _group_by_paper(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bucket chunks by paper_id, sort by retrieval score (best first within a
    paper), and emit one dict per paper carrying the metadata + chunks list.
    Papers are ordered by their best chunk score."""
    buckets: dict[str, dict[str, Any]] = {}
    for c in chunks:
        pid = c["paper_id"]
        if pid not in buckets:
            buckets[pid] = {
                "paper_id": pid,
                "title": c["title"],
                "year": c["year"],
                "authors": c["authors"],
                "doi": c["doi"],
                "pdf_url": c["pdf_url"],
                "domain": c["domain"],
                "field": c["field"],
                "subfield": c["subfield"],
                "citations": c["citations"],
                "chunks": [],
                "_best_score": c["score"],
            }
        buckets[pid]["chunks"].append(
            {
                "chunk_id": c["chunk_id"],
                "section_title": c["section_title"],
                "section_type": c["section_type"],
                "position": c["position"],
                "text": c["text"],
                "score": c["score"],
            }
        )
        if c["score"] > buckets[pid]["_best_score"]:
            buckets[pid]["_best_score"] = c["score"]

    papers = sorted(buckets.values(), key=lambda p: p["_best_score"], reverse=True)
    for p in papers:
        p["chunks"].sort(key=lambda c: c["score"], reverse=True)
        p.pop("_best_score", None)
    return papers
