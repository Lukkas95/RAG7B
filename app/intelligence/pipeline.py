"""End-to-end intelligence pipelines.

Three orchestrators — gaps, ToC, methodologies — share the same preamble:

  user query
    -> expand_query()                                 (LLM)
    -> hybrid_search per variant, biased to the pipeline's analytical sections   (DB)
    -> fallback unfiltered pass if too few distinct papers found
    -> group chunks by paper, attach paper metadata

…and differ only in (a) which `section_types` filter the retrieval uses and
(b) which synthesis workflow they hand the grouped papers to. The shared
preamble lives in `_collect_papers`; each `run_*_pipeline` wrapper is a few
lines on top.

`run_pipeline` is kept as a back-compat alias for `run_gaps_pipeline` so
`scripts/run_pipeline.py` keeps working unchanged.
"""
import os
from functools import partial
from typing import Any, Awaitable, Callable, Optional

import asyncio

from app.intelligence.llm import describe_backend
from app.intelligence.workflows import (
    expand_query,
    gap_synthesis,
    general_expand_query,
    methodology_synthesis,
    qa_synthesis,
    toc_synthesis,
)
from app.retrieval import hybrid_search

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000").rstrip("/")

# Section-type filters per pipeline.
GAPS_SECTION_TYPES: list[str] = ["limitation", "discussion", "conclusion"]
# unfiltered — outline needs all sections
TOC_SECTION_TYPES: Optional[list[str]] = None
METHODOLOGY_SECTION_TYPES: list[str] = ["method", "result"]
# qa runs unfiltered — answers can live in any section.
QA_SECTION_TYPES: Optional[list[str]] = None

# Back-compat: older code may still import this name.
ANALYTICAL_SECTION_TYPES = GAPS_SECTION_TYPES


async def _collect_papers(
    user_query: str,
    *,
    top_k_per_query: int,
    min_distinct_papers: int,
    section_types: Optional[list[str]],
    log,
    expander: Callable[[str], Awaitable[list[str]]] = expand_query,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Shared preamble for all retrieval pipelines:

      expander(user_query) → fan-out hybrid_search(section_types=…) →
      dedupe → fallback unfiltered if <min_distinct_papers → group_by_paper.

    The three button pipelines use the default `expand_query` expander; the
    qa pipeline passes `general_expand_query` for a neutral rewrite.
    Returns (expanded_queries, papers).
    """
    log("[pipeline] expanding query...")
    expanded = await expander(user_query)
    log(f"[pipeline] {len(expanded)} expanded queries: {expanded}")

    log(f"[pipeline] retrieving (filtered to {section_types})...")
    chunk_lists = await asyncio.gather(
        *[
            hybrid_search(q, top_k=top_k_per_query,
                          section_types=section_types)
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
        chunks = _dedupe_chunks(
            chunks + [c for lst in fallback_lists for c in lst])
        log(
            f"[pipeline] after fallback: {len(chunks)} chunks across "
            f"{len({c['paper_id'] for c in chunks})} papers"
        )

    papers = _group_by_paper(chunks)
    log(f"[pipeline] grouped into {len(papers)} papers")
    return expanded, papers


async def _run(
    user_query: str,
    *,
    section_types: Optional[list[str]],
    synthesizer: Callable[[list[dict[str, Any]]], Awaitable[str]],
    pipeline_label: str,
    top_k_per_query: int,
    min_distinct_papers: int,
    verbose: bool,
    expander: Callable[[str], Awaitable[list[str]]] = expand_query,
) -> dict[str, Any]:
    log = print if verbose else (lambda *a, **k: None)
    log(f"[pipeline:{pipeline_label}] backend={describe_backend()} | query={user_query!r}")

    expanded, papers = await _collect_papers(
        user_query,
        top_k_per_query=top_k_per_query,
        min_distinct_papers=min_distinct_papers,
        section_types=section_types,
        log=log,
        expander=expander,
    )

    log(f"[pipeline:{pipeline_label}] calling synthesis LLM...")
    analysis = await synthesizer(papers)
    log(f"[pipeline:{pipeline_label}] done.")

    return {
        "query": user_query,
        "expanded_queries": expanded,
        "papers": papers,
        "analysis": analysis,
    }


async def run_gaps_pipeline(
    user_query: str,
    *,
    top_k_per_query: int = 8,
    min_distinct_papers: int = 3,
    verbose: bool = True,
) -> dict[str, Any]:
    """Cross-paper limitations / future-work / research-silence analysis."""
    return await _run(
        user_query,
        section_types=GAPS_SECTION_TYPES,
        synthesizer=gap_synthesis,
        pipeline_label="gaps",
        top_k_per_query=top_k_per_query,
        min_distinct_papers=min_distinct_papers,
        verbose=verbose,
    )


async def run_toc_pipeline(
    user_query: str,
    *,
    top_k_per_query: int = 8,
    min_distinct_papers: int = 3,
    verbose: bool = True,
) -> dict[str, Any]:
    """Hierarchical Table-of-Contents for a discussion across the papers."""
    return await _run(
        user_query,
        section_types=TOC_SECTION_TYPES,
        synthesizer=toc_synthesis,
        pipeline_label="toc",
        top_k_per_query=top_k_per_query,
        min_distinct_papers=min_distinct_papers,
        verbose=verbose,
    )


async def run_methodologies_pipeline(
    user_query: str,
    *,
    top_k_per_query: int = 8,
    min_distinct_papers: int = 3,
    verbose: bool = True,
) -> dict[str, Any]:
    """Per-paper methodology profile + cross-paper comparative matrix."""
    return await _run(
        user_query,
        section_types=METHODOLOGY_SECTION_TYPES,
        synthesizer=methodology_synthesis,
        pipeline_label="methodologies",
        top_k_per_query=top_k_per_query,
        min_distinct_papers=min_distinct_papers,
        verbose=verbose,
    )


async def run_qa_pipeline(
    user_query: str,
    *,
    top_k_per_query: int = 8,
    min_distinct_papers: int = 0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Grounded general Q&A. Runs unfiltered retrieval with a neutral query
    expansion, then asks the LLM to answer the user's actual question using
    only the retrieved chunks. Used as the chat router's default for any
    research question that isn't an explicit gaps/toc/methodologies ask.

    `min_distinct_papers=0` because the unfiltered fallback in _collect_papers
    would just re-run the same (already unfiltered) searches — no point.
    """
    return await _run(
        user_query,
        section_types=QA_SECTION_TYPES,
        synthesizer=partial(qa_synthesis, user_query),
        pipeline_label="qa",
        top_k_per_query=top_k_per_query,
        min_distinct_papers=min_distinct_papers,
        verbose=verbose,
        expander=general_expand_query,
    )


# Back-compat alias — older callers (e.g. `scripts/run_pipeline.py`) still
# import `run_pipeline`.
run_pipeline = run_gaps_pipeline


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
                "pdf": f"{SERVER_URL}/chunks/{c['chunk_id']}/pdf",
                "section_title": c["section_title"],
                "section_type": c["section_type"],
                "position": c["position"],
                "text": c["text"],
                "score": c["score"],
            }
        )
        if c["score"] > buckets[pid]["_best_score"]:
            buckets[pid]["_best_score"] = c["score"]

    papers = sorted(buckets.values(),
                    key=lambda p: p["_best_score"], reverse=True)
    for p in papers:
        p["chunks"].sort(key=lambda c: c["score"], reverse=True)
        p.pop("_best_score", None)
    return papers
