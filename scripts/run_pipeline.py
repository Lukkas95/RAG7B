"""CLI entry point for the intelligence pipeline.

Usage:
    python3 scripts/run_pipeline.py "your question"

Env vars:
    LLM_BACKEND     gemini (default) | ollama | openrouter
    LLM_MODEL       overrides default model id for the chosen backend
    GOOGLE_API_KEY  required if LLM_BACKEND=gemini
    OLLAMA_HOST     default http://localhost:11434
    OPENROUTER_API_KEY required if LLM_BACKEND=openrouter
    DATABASE_URL    default postgresql://rag:rag@localhost:5432/scholargraph
"""
import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Allow `python scripts/run_pipeline.py ...` from the repo root by adding the
# repo root to sys.path so `import app...` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from app.db import close_pool, init_pool  # noqa: E402
from app.embeddings import get_model  # noqa: E402
from app.intelligence.pipeline import run_pipeline  # noqa: E402


async def _main(query: str, top_k: int, output_dir: Path) -> None:
    await init_pool()
    try:
        get_model()  # preload sentence-transformers
        result = await run_pipeline(query, top_k_per_query=top_k)
    finally:
        await close_pool()

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print(result["analysis"])

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_query = "".join(c if c.isalnum() else "_" for c in query)[:60]

    md_path = output_dir / f"{stamp}_{safe_query}.md"
    md_path.write_text(_render_markdown(result), encoding="utf-8")

    json_path = output_dir / f"{stamp}_{safe_query}.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\nSaved: {md_path}")
    print(f"Saved: {json_path}")


def _render_markdown(result: dict) -> str:
    lines = [
        f"# {result['query']}",
        "",
        "## Expanded queries",
        *[f"- {q}" for q in result["expanded_queries"]],
        "",
        f"## Papers retrieved ({len(result['papers'])})",
    ]
    for p in result["papers"]:
        authors = ", ".join((p.get("authors") or [])[:3])
        if p.get("authors") and len(p["authors"]) > 3:
            authors += " et al."
        lines += [
            f"- **{p['title']}** ({p.get('year', 'n.d.')}) — {authors or 'unknown'}",
            f"  - {len(p['chunks'])} chunks; sections: "
            + ", ".join(sorted({c["section_type"] for c in p["chunks"]})),
        ]
    lines += ["", "## Analysis", "", result["analysis"]]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="The question to investigate")
    parser.add_argument(
        "--top-k", type=int, default=8, help="Chunks retrieved per expanded query"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Where to write the report (default: ./output)",
    )
    args = parser.parse_args()
    asyncio.run(_main(args.query, args.top_k, args.output_dir))


if __name__ == "__main__":
    main()
