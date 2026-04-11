from typing import List

from fastapi import APIRouter, HTTPException

from app.db import get_pool
from app.models import DocumentCreate, DocumentResponse

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentResponse, status_code=201)
async def create_document(doc: DocumentCreate):
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO documents (title, authors, year, source)
            VALUES ($1, $2, $3, $4)
            RETURNING *
            """,
            doc.title, doc.authors, doc.year, doc.source,
        )
    return dict(row)


@router.get("", response_model=List[DocumentResponse])
async def list_documents():
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM documents ORDER BY created_at DESC"
        )
    return [dict(r) for r in rows]
