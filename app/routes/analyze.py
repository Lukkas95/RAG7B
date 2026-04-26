"""End-to-end intelligence pipeline as an HTTP endpoint.

Thin wrapper around `app.intelligence.pipeline.run_pipeline` so the frontend
can hit one URL and get back expanded queries, retrieved papers, and the
gap-synthesis analysis in a single response.
"""
from fastapi import APIRouter

from app.intelligence.pipeline import run_pipeline
from app.models import AnalyzeRequest, AnalyzeResponse

router = APIRouter(tags=["analyze"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    result = await run_pipeline(
        req.query,
        top_k_per_query=req.top_k_per_query,
        verbose=False,
    )
    return AnalyzeResponse(**result)
