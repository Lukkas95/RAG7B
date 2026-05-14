"""Intent-routed chat endpoint.

`POST /chat` accepts the full conversation. `decide(messages)` returns both
(a) one of {gaps, toc, methodologies, qa, text} and (b) a self-contained
`search_query` rewritten from the conversation (resolves anaphora /
follow-ups). The route then dispatches:

  - "gaps"          → run_gaps_pipeline(search_query)
  - "toc"           → run_toc_pipeline(search_query)
  - "methodologies" → run_methodologies_pipeline(search_query)
  - "qa"            → run_qa_pipeline(search_query)   (default for any
                       grounded research question that isn't an explicit
                       gaps/toc/methodologies ask)
  - "text"          → text_response(messages)   (LLM-only, no DB hit;
                       search_query is unused on this branch)

The rewritten `search_query` is echoed back in `ChatResponse.query` so the
frontend can show the user what was actually retrieved against.
"""
from fastapi import APIRouter

from app.intelligence.router import PIPELINES, decide, text_response
from app.models import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    messages = [m.model_dump() for m in req.messages]
    pipeline, search_query = await decide(messages)

    if pipeline == "text":
        analysis = await text_response(messages)
        return ChatResponse(pipeline="text", analysis=analysis)

    pipeline_fn = PIPELINES[pipeline]
    result = await pipeline_fn(
        search_query,
        top_k_per_query=req.top_k_per_query,
        verbose=False,
    )
    return ChatResponse(pipeline=pipeline, **result)
