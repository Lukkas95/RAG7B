"""Reasoning workflows: query expansion + the three button-triggered syntheses.

All call into `app.intelligence.llm.complete()`, so they're agnostic to which
LLM backend is configured. Each synthesis takes the same per-paper input shape
(produced by `pipeline._group_by_paper`) and returns a string analysis.
"""
import json
import re
from typing import Any

from app.intelligence.llm import complete
from app.intelligence.prompts import (
    EXPANSION_PROMPT_TEMPLATE,
    GAP_SYNTHESIS_PROMPT_TEMPLATE,
    METHODOLOGY_SYNTHESIS_PROMPT_TEMPLATE,
    QA_EXPANSION_PROMPT_TEMPLATE,
    QA_SYNTHESIS_PROMPT_TEMPLATE,
    TOC_SYNTHESIS_PROMPT_TEMPLATE,
)


async def expand_query(user_query: str) -> list[str]:
    """Turn one user query into 3 technical variations. Falls back to
    `[user_query]` if the LLM output can't be parsed as a JSON list."""
    return await _expand(user_query, EXPANSION_PROMPT_TEMPLATE)


async def general_expand_query(user_query: str) -> list[str]:
    """Neutral variant of `expand_query` used by the qa pipeline. Same JSON
    contract; the prompt does NOT bias toward methodology / results /
    constraints, so it's a better fit for general grounded Q&A."""
    return await _expand(user_query, QA_EXPANSION_PROMPT_TEMPLATE)


async def gap_synthesis(papers: list[dict[str, Any]]) -> str:
    """Cross-paper limitations / future-work / silence analysis."""
    return await _synthesize(papers, GAP_SYNTHESIS_PROMPT_TEMPLATE)


async def toc_synthesis(papers: list[dict[str, Any]]) -> str:
    """Hierarchical Table-of-Contents for a discussion synthesizing the papers."""
    return await _synthesize(papers, TOC_SYNTHESIS_PROMPT_TEMPLATE)


async def methodology_synthesis(papers: list[dict[str, Any]]) -> str:
    """Per-paper methodology profile + cross-paper comparative matrix."""
    return await _synthesize(papers, METHODOLOGY_SYNTHESIS_PROMPT_TEMPLATE)


async def qa_synthesis(user_query: str, papers: list[dict[str, Any]]) -> str:
    """Answer the user's actual question grounded in the provided papers.

    Unlike the three fixed-task syntheses above, the qa prompt needs the
    user's question as input (alongside the context), so this can't reuse
    `_synthesize` — it formats {user_query} + {context_data} into
    QA_SYNTHESIS_PROMPT_TEMPLATE directly.
    """
    if not papers:
        return "Information not available in the provided sources."
    context = "\n\n---\n\n".join(_format_paper(p) for p in papers)
    prompt = QA_SYNTHESIS_PROMPT_TEMPLATE.format(
        user_query=user_query, context_data=context
    )
    return await complete(prompt)


async def _expand(user_query: str, template: str) -> list[str]:
    prompt = template.format(user_query=user_query)
    raw = await complete(prompt)
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return [user_query]
    try:
        parsed = json.loads(match.group(0))
        return [str(q) for q in parsed if isinstance(q, str)] or [user_query]
    except (json.JSONDecodeError, TypeError):
        return [user_query]


async def _synthesize(papers: list[dict[str, Any]], template: str) -> str:
    """Shared body of all three synthesis workflows: format papers into the
    [Context Data] block, fill the template, run a single `complete()` call.

    Each entry in `papers` must carry at minimum `title`, `year`, and `chunks`
    (a list of dicts with `section_title`, `section_type`, `position`, `text`).
    """
    if not papers:
        return "Information not available in the provided sources."
    context = "\n\n---\n\n".join(_format_paper(p) for p in papers)
    prompt = template.format(context_data=context)
    return await complete(prompt)


def _format_paper(paper: dict[str, Any]) -> str:
    title = paper.get("title", "Unknown Paper")
    year = paper.get("year", "n.d.")
    authors = paper.get("authors") or []
    authors_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
    field = paper.get("subfield") or paper.get("field") or "n/a"
    citations = paper.get("citations", 0)
    doi = paper.get("doi") or ""

    header = (
        f"# {title} ({year})\n"
        f"  Authors: {authors_str or 'unknown'}\n"
        f"  Field: {field} | Citations: {citations} | DOI: {doi}\n"
        f"  PDF: {paper.get('pdf_url') or 'n/a'}\n"
        f"  Chunks ({len(paper.get('chunks', []))}):"
    )
    chunk_blocks = []
    for c in paper.get("chunks", []):
        chunk_blocks.append(
            f"  [§{c.get('section_title', '?')} | type={c.get('section_type', '?')} "
            f"| pos={c.get('position', '?')}]\n"
            f"  {c.get('text', '').strip()}"
        )
    return header + "\n" + "\n\n".join(chunk_blocks)
