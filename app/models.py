from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field


# --- Request models ---

class ChunkIngest(BaseModel):
    """Matches the JSONL format from Role A (chunked_results.jsonl)."""
    chunk_id: Optional[str] = None
    text: str
    section_title: Optional[str] = None
    section_type: Optional[str] = None
    section_index: Optional[int] = None
    chunk_index: Optional[int] = None
    position: int
    is_abstract: bool = False
    paper_id: str
    title: str
    year: Optional[int] = None
    authors: List[str] = []
    doi: Optional[str] = None
    domain: Optional[str] = None
    field: Optional[str] = None
    subfield: Optional[str] = None
    citations: int = 0


class ChunkBatchIngest(BaseModel):
    chunks: List[ChunkIngest] = Field(max_length=500)


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    semantic_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    section_type: Optional[str] = None
    domain: Optional[str] = None
    field: Optional[str] = None
    subfield: Optional[str] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    paper_id: Optional[str] = None
    is_abstract: Optional[bool] = None


# --- Response models ---

class ChunkResponse(BaseModel):
    id: str
    position: int
    text: str
    section_title: Optional[str]
    section_type: Optional[str]
    section_index: Optional[int]
    chunk_index: Optional[int]
    is_abstract: bool
    paper_id: str
    title: str
    year: Optional[int]
    authors: List[str]
    doi: Optional[str]
    domain: Optional[str]
    field: Optional[str]
    subfield: Optional[str]
    citations: int
    created_at: datetime


class SearchResult(BaseModel):
    chunk_id: str
    position: int
    text: str
    section_title: Optional[str]
    section_type: Optional[str]
    is_abstract: bool
    paper_id: str
    title: str
    year: Optional[int]
    authors: List[str]
    doi: Optional[str]
    domain: Optional[str]
    score: float


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]


class ContextResponse(BaseModel):
    target: ChunkResponse
    before: Optional[ChunkResponse]
    after: Optional[ChunkResponse]
