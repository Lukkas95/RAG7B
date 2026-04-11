"""
Load the rag-mini-wikipedia demo dataset into ScholarGraph via the API.

Groups every 20 consecutive passages into a synthetic "document" and
posts them as chunks through the API endpoints.
"""

import httpx
from datasets import load_dataset

API_BASE = "http://localhost:8000"
BATCH_SIZE = 20  # passages per synthetic document


def main():
    print("Loading dataset from HuggingFace...")
    ds = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus", split="passages")
    passages = list(ds)
    total = len(passages)
    print(f"Loaded {total} passages. Grouping into documents of {BATCH_SIZE}...")

    client = httpx.Client(timeout=120.0)
    docs_created = 0
    chunks_inserted = 0

    for start in range(0, total, BATCH_SIZE):
        batch = passages[start : start + BATCH_SIZE]
        doc_num = start // BATCH_SIZE + 1

        # Create a synthetic document
        resp = client.post(f"{API_BASE}/documents", json={
            "title": f"Wiki Collection {doc_num}",
            "source": "rag-datasets/rag-mini-wikipedia",
        })
        resp.raise_for_status()
        doc_id = resp.json()["id"]
        docs_created += 1

        # Post chunks
        chunks = [
            {
                "chunk_index": i,
                "content": p["passage"],
            }
            for i, p in enumerate(batch)
        ]
        resp = client.post(f"{API_BASE}/documents/{doc_id}/chunks", json={"chunks": chunks})
        resp.raise_for_status()
        inserted = resp.json()["inserted"]
        chunks_inserted += inserted
        print(f"  Doc {doc_num}: {inserted} chunks (total: {chunks_inserted}/{total})")

    print(f"\nDone! Created {docs_created} documents, {chunks_inserted} chunks total.")


if __name__ == "__main__":
    main()
