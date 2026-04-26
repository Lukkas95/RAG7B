import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db import init_pool, close_pool, get_pool
from app.embeddings import get_model
from app.routes import analyze, chunks, search


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

# CORS — allow the frontend (on a different port) to call this API.
# Override via CORS_ORIGINS env var (comma-separated). "*" allows everything.
_origins_env = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://localhost:8080",
)
_allow_origins = [o.strip() for o in _origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chunks.router)
app.include_router(search.router)
app.include_router(analyze.router)


@app.get("/health")
async def health():
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.fetchval("SELECT 1")
    return {"status": "ok"}
