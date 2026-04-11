from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.db import init_pool, close_pool, get_pool
from app.embeddings import get_model
from app.routes import documents, chunks, search


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_pool()
    get_model()  # preload embedding model on startup
    yield
    await close_pool()


app = FastAPI(
    title="ScholarGraph RAG",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(documents.router)
app.include_router(chunks.router)
app.include_router(search.router)


@app.get("/health")
async def health():
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.fetchval("SELECT 1")
    return {"status": "ok"}
