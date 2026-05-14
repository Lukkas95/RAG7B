"""Chat-side intent routing.

`/chat` accepts the full conversation, calls `decide(messages)` to (a)
classify the user's intent and (b) rewrite the conversation into a
self-contained search query, then dispatches to one of the three retrieval
pipelines or to a `text_response()` fallback (LLM-only, no DB hit).

`decide` formats the conversation into a classifier prompt
(`INTENT_CLASSIFIER_PROMPT_TEMPLATE`), asks the LLM for a two-field JSON
object `{"pipeline": ..., "search_query": ...}`, and returns
`(label, search_query)`. The rewrite resolves anaphora ("What about the
methods?" → carries forward the topic from earlier turns) and strips
conversational filler so the retriever doesn't waste cosine budget on
"could you please".

Fallbacks (any parse failure or unknown label):
- `pipeline` → ``"text"`` — safest "I don't know" answer, produces a plain
  conversational reply rather than a malformed paper analysis.
- `search_query` → `last_user_message(messages)` — preserves the previous
  behaviour where retrieval ran on the latest user turn verbatim.

Lives in its own module (rather than inside `workflows.py`) because it
imports the three pipeline orchestrators from `pipeline.py`, which already
imports `workflows.py` — putting `decide` in `workflows.py` would create a
circular import.
"""
import json
import re
from typing import Any, Awaitable, Callable

from app.intelligence.llm import complete
from app.intelligence.pipeline import (
    run_gaps_pipeline,
    run_methodologies_pipeline,
    run_qa_pipeline,
    run_toc_pipeline,
)
from app.intelligence.prompts import INTENT_CLASSIFIER_PROMPT_TEMPLATE

# Labels match the `pipeline` field in ChatResponse.
PipelineLabel = str  # one of: "gaps" | "toc" | "methodologies" | "qa" | "text"
VALID_LABELS: set[str] = {"gaps", "toc", "methodologies", "qa", "text"}
DEFAULT_LABEL: PipelineLabel = "text"

PIPELINES: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {
    "gaps": run_gaps_pipeline,
    "toc": run_toc_pipeline,
    "methodologies": run_methodologies_pipeline,
    "qa": run_qa_pipeline,
}


async def decide(
    messages: list[dict[str, Any]],
) -> tuple[PipelineLabel, str]:
    """Classify intent AND rewrite the conversation into a search query.

    Hands the whole conversation to the LLM and asks for a two-field JSON
    object `{"pipeline": ..., "search_query": ...}`. The rewrite carries
    forward earlier topic context so a follow-up like "What about the
    methods?" still produces a useful retrieval query. Returns
    ``(label, search_query)`` — fallbacks are independent so a partial
    LLM response (good label, missing query) still works.
    """
    fallback_query = last_user_message(messages)
    if not messages:
        return DEFAULT_LABEL, fallback_query

    prompt = INTENT_CLASSIFIER_PROMPT_TEMPLATE.format(
        conversation=_format_conversation(messages)
    )
    raw = await complete(prompt)

    # Pull the first {...} block out of the raw response (handles stray
    # commentary, code fences, leading/trailing whitespace).
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return DEFAULT_LABEL, fallback_query
    try:
        parsed = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return DEFAULT_LABEL, fallback_query
    if not isinstance(parsed, dict):
        return DEFAULT_LABEL, fallback_query

    raw_label = parsed.get("pipeline")
    if isinstance(raw_label, str) and raw_label.strip().lower() in VALID_LABELS:
        label: PipelineLabel = raw_label.strip().lower()
    else:
        label = DEFAULT_LABEL

    raw_query = parsed.get("search_query")
    if isinstance(raw_query, str) and raw_query.strip():
        search_query = raw_query.strip()
    else:
        search_query = fallback_query

    return label, search_query


def _format_conversation(messages: list[dict[str, Any]]) -> str:
    """Render the chat history into a stable, role-prefixed transcript.
    Keeps the tail of the conversation prominent — the classifier should
    weight the most recent user turn most heavily."""
    return "\n".join(
        f"{(m.get('role') or 'user').upper()}: {(m.get('content') or '').strip()}"
        for m in messages
    )


async def text_response(messages: list[dict[str, Any]]) -> str:
    """Pure-LLM fallback used when `decide` returns ``"text"`` — no retrieval,
    no DB hit. Hands the full conversation to the LLM and returns its reply.
    """
    if not messages:
        return ""
    convo = "\n".join(
        f"{(m.get('role') or 'user').upper()}: {m.get('content', '')}"
        for m in messages
    )
    prompt = (
        "You are a helpful research assistant. Continue the conversation, "
        "answering the user's last message. Be concise and stay on topic.\n\n"
        f"{convo}\n\nASSISTANT:"
    )
    return await complete(prompt)


def last_user_message(messages: list[dict[str, Any]]) -> str:
    """Extract the latest user-authored message; empty string if none."""
    for m in reversed(messages):
        if (m.get("role") or "").lower() == "user":
            return m.get("content", "") or ""
    return ""
