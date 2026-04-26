from fastapi import APIRouter

from app.models import SearchRequest, SearchResponse, SearchResult
from app.retrieval import hybrid_search

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    section_types = [req.section_type] if req.section_type else None
    rows = await hybrid_search(
        req.query,
        top_k=req.top_k,
        semantic_weight=req.semantic_weight,
        keyword_weight=req.keyword_weight,
        section_types=section_types,
        domain=req.domain,
        field=req.field,
        subfield=req.subfield,
        paper_id=req.paper_id,
        year_min=req.year_min,
        year_max=req.year_max,
        is_abstract=req.is_abstract,
    )
    results = [SearchResult(**r) for r in rows]
    return SearchResponse(query=req.query, results=results)
