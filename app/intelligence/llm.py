"""Pluggable LLM backend.

Chosen via env var `LLM_BACKEND` (default `gemini`):
  - `gemini` — Google `google.genai` SDK; requires `GOOGLE_API_KEY`.
  - `ollama` — local Ollama server at `OLLAMA_HOST` (default
    http://localhost:11434). Run `ollama pull <model>` first.
  - `openrouter` — OpenRouter Python SDK; requires `OPENROUTER_API_KEY`.

Override the model id via `LLM_MODEL`.
"""
import asyncio
import os
from functools import lru_cache

import httpx


def _backend() -> str:
    return os.getenv("LLM_BACKEND", "gemini").lower()


def _model_id() -> str:
    backend = _backend()
    default = {
        "gemini": "gemini-flash-latest",
        "ollama": "qwen2.5:7b-instruct",
        "openrouter": "qwen/qwen-2.5-7b-instruct",
    }.get(backend, "gemini-flash-latest")
    return os.getenv("LLM_MODEL", default)


@lru_cache(maxsize=1)
def _gemini_client():
    from google import genai  # imported lazily so ollama-only users don't need it

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not set (required for LLM_BACKEND=gemini)")
    return genai.Client(api_key=api_key)


async def _complete_gemini(prompt: str) -> str:
    client = _gemini_client()
    model = _model_id()

    def _call() -> str:
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text

    return await asyncio.to_thread(_call)


async def _complete_ollama(prompt: str) -> str:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = _model_id()
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        r = await client.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
        )
        r.raise_for_status()
        return r.json()["response"]


async def _complete_openrouter(prompt: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set (required for LLM_BACKEND=openrouter)")

    model = _model_id()

    # imported lazily so non-openrouter users don't need it
    from openrouter import OpenRouter

    def _call() -> str:
        with OpenRouter(api_key=api_key) as client:
            response = client.chat.send(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content

    return await asyncio.to_thread(_call)


async def complete(prompt: str) -> str:
    """Run a single-turn completion through the configured backend."""
    backend = _backend()
    print(f"Completing with {describe_backend()}...")
    if backend == "gemini":
        return await _complete_gemini(prompt)
    if backend == "ollama":
        return await _complete_ollama(prompt)
    if backend == "openrouter":
        return await _complete_openrouter(prompt)
    raise ValueError(
        f"Unknown LLM_BACKEND: {backend!r} (expected 'gemini', 'ollama', or 'openrouter')"
    )


def describe_backend() -> str:
    """Human-readable string for logging."""
    return f"{_backend()}:{_model_id()}"
