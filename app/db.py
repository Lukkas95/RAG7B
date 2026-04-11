import os
from typing import Optional

import asyncpg
from pgvector.asyncpg import register_vector

pool: Optional[asyncpg.Pool] = None

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://rag:rag@localhost:5432/scholargraph",
)


async def _init_connection(conn: asyncpg.Connection):
    """Register the pgvector type codec on every new pool connection."""
    await register_vector(conn)


async def init_pool():
    global pool
    pool = await asyncpg.create_pool(
        dsn=DATABASE_URL,
        min_size=2,
        max_size=10,
        init=_init_connection,
    )


async def close_pool():
    global pool
    if pool:
        await pool.close()
        pool = None


def get_pool() -> asyncpg.Pool:
    if pool is None:
        raise RuntimeError("Database pool not initialized")
    return pool
