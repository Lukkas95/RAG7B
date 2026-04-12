"""
Load the rag-mini-wikipedia demo dataset into ScholarGraph via the API.

Synthesizes paper-level metadata from the wiki passages to match
the new ChunkIngest format.
"""

import uuid

import httpx
from datasets import load_dataset

API_BASE = "http://localhost:8000"
BATCH_SIZE = 50  # chunks per API call


def main():
    print("Loading dataset from HuggingFace...")
    ds = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus", split="passages")
    passages = list(ds)
    total = len(passages)
    print(f"Loaded {total} passages. Posting in batches of {BATCH_SIZE}...")

    client = httpx.Client(timeout=120.0)
    chunks_inserted = 0

    # Group every 20 passages into a synthetic "paper"
    paper_size = 20

    for start in range(0, total, BATCH_SIZE):
        batch = passages[start : start + BATCH_SIZE]

        chunks = []
        for i, p in enumerate(batch):
            global_idx = start + i
            paper_num = global_idx // paper_size
            position = global_idx % paper_size

            chunks.append({
                "chunk_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"wiki-{p['id']}")),
                "text": p["passage"],
                "section_title": None,
                "section_type": None,
                "position": position,
                "paper_id": f"wiki-collection-{paper_num}",
                "title": f"Wiki Collection {paper_num + 1}",
                "year": None,
                "authors": [],
                "venue": None,
                "domain": "General Knowledge",
                "field": "Wikipedia",
                "subfield": None,
                "topics": [],
                "citations": 0,
            })

        resp = client.post(f"{API_BASE}/chunks", json={"chunks": chunks})
        resp.raise_for_status()
        inserted = resp.json()["inserted"]
        chunks_inserted += inserted
        print(f"  Batch: {inserted} chunks (total: {chunks_inserted}/{total})")

    print(f"\nDone! Inserted {chunks_inserted} chunks total.")


if __name__ == "__main__":
    main()
