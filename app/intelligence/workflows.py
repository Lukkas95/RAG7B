"""Reasoning workflows: query expansion + gap synthesis.

Both call into `app.intelligence.llm.complete()`, so they're agnostic to which
LLM backend is configured.
"""
import json
import re
from typing import Any

from app.intelligence.llm import complete
from app.intelligence.prompts import (
    EXPANSION_PROMPT_TEMPLATE,
    GAP_SYNTHESIS_PROMPT_TEMPLATE,
)


async def expand_query(user_query: str) -> list[str]:
    """Turn one user query into 3 technical variations. Falls back to
    `[user_query]` if the LLM output can't be parsed as a JSON list."""
    prompt = EXPANSION_PROMPT_TEMPLATE.format(user_query=user_query)
    raw = await complete(prompt)
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return [user_query]
    try:
        parsed = json.loads(match.group(0))
        return [str(q) for q in parsed if isinstance(q, str)] or [user_query]
    except (json.JSONDecodeError, TypeError):
        return [user_query]


async def gap_synthesis(papers: list[dict[str, Any]]) -> str:
    """Cross-paper analysis. Each entry in `papers` must carry at minimum
    `title`, `year`, and `chunks` (a list of dicts with `section_title`,
    `section_type`, `position`, `text`)."""
    if not papers:
        return "Information not available in the provided sources."
    context = "\n\n---\n\n".join(_format_paper(p) for p in papers)
    prompt = GAP_SYNTHESIS_PROMPT_TEMPLATE.format(context_data=context)
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
