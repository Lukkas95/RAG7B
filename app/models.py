from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field


# --- Request models ---

class DocumentCreate(BaseModel):
    title: str
    authors: Optional[str] = None
    year: Optional[int] = None
    source: Optional[str] = None


class ChunkCreate(BaseModel):
    chunk_index: int
    content: str
    section_label: Optional[str] = None


class ChunkBatchCreate(BaseModel):
    chunks: List[ChunkCreate] = Field(max_length=500)


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    semantic_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    section_label: Optional[str] = None


# --- Response models ---

class DocumentResponse(BaseModel):
    id: int
    title: str
    authors: Optional[str]
    year: Optional[int]
    source: Optional[str]
    created_at: datetime


class ChunkResponse(BaseModel):
    id: int
    document_id: int
    chunk_index: int
    content: str
    section_label: Optional[str]
    created_at: datetime


class SearchResult(BaseModel):
    chunk_id: int
    document_id: int
    chunk_index: int
    content: str
    section_label: Optional[str]
    score: float
    document_title: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]


class ContextResponse(BaseModel):
    target: ChunkResponse
    before: Optional[ChunkResponse]
    after: Optional[ChunkResponse]
