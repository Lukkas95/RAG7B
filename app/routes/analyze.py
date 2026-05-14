"""Three button-triggered pipelines + a back-compat alias.

Each route is a thin wrapper around an orchestrator in
`app.intelligence.pipeline`. They all return the same `AnalyzeResponse`
shape so the frontend can render them uniformly.

| Endpoint                     | Orchestrator                | Section filter            |
| ---------------------------- | --------------------------- | ------------------------- |
| POST /analyze/gaps           | run_gaps_pipeline           | limitation/discussion/conclusion |
| POST /analyze/toc            | run_toc_pipeline            | unfiltered                |
| POST /analyze/methodologies  | run_methodologies_pipeline  | method/result             |
| POST /analyze/qa             | run_qa_pipeline             | unfiltered                |
| POST /analyze (alias)        | run_gaps_pipeline           | (= /analyze/gaps)         |
"""
from fastapi import APIRouter

from app.intelligence.pipeline import (
    run_gaps_pipeline,
    run_methodologies_pipeline,
    run_qa_pipeline,
    run_toc_pipeline,
)
from app.models import AnalyzeRequest, AnalyzeResponse

router = APIRouter(tags=["analyze"])


@router.post("/analyze/gaps", response_model=AnalyzeResponse)
async def analyze_gaps(req: AnalyzeRequest) -> AnalyzeResponse:
    """Cross-paper limitations / future-work / research-silence analysis."""
    result = await run_gaps_pipeline(
        req.query,
        top_k_per_query=req.top_k_per_query,
        verbose=False,
    )
    return AnalyzeResponse(**result)


@router.post("/analyze/toc", response_model=AnalyzeResponse)
async def analyze_toc(req: AnalyzeRequest) -> AnalyzeResponse:
    """Hierarchical Table-of-Contents for a discussion across the papers."""
    result = await run_toc_pipeline(
        req.query,
        top_k_per_query=req.top_k_per_query,
        verbose=False,
    )
    return AnalyzeResponse(**result)


@router.post("/analyze/methodologies", response_model=AnalyzeResponse)
async def analyze_methodologies(req: AnalyzeRequest) -> AnalyzeResponse:
    """Per-paper methodology profile + cross-paper comparative matrix."""
    result = await run_methodologies_pipeline(
        req.query,
        top_k_per_query=req.top_k_per_query,
        verbose=False,
    )
    return AnalyzeResponse(**result)


@router.post("/analyze/qa", response_model=AnalyzeResponse)
async def analyze_qa(req: AnalyzeRequest) -> AnalyzeResponse:
    """Grounded general Q&A — unfiltered retrieval, answers the user's
    actual question using only retrieved chunks. Also used as the chat
    router's default for any research question that isn't an explicit
    gaps/toc/methodologies ask."""
    result = await run_qa_pipeline(
        req.query,
        top_k_per_query=req.top_k_per_query,
        verbose=False,
    )
    return AnalyzeResponse(**result)


@router.post("/analyze", response_model=AnalyzeResponse, deprecated=True)
async def analyze_legacy(req: AnalyzeRequest) -> AnalyzeResponse:
    """Back-compat alias for `POST /analyze/gaps`. Pre-existing frontends
    (and any external scripts) keep working without changes; new callers
    should hit `/analyze/gaps` directly."""
    result = await run_gaps_pipeline(
        req.query,
        top_k_per_query=req.top_k_per_query,
        verbose=False,
    )
    return AnalyzeResponse(**result)
