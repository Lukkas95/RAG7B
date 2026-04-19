"""
Load a chunked_results.jsonl file into ScholarGraph via the API.

Usage:
    python3 scripts/load_jsonl.py chunked_results.jsonl
    python3 scripts/load_jsonl.py path/to/any_file.jsonl --batch-size 200
"""

import argparse
import json
import time

import httpx

API_BASE = "http://localhost:8000"
DEFAULT_BATCH_SIZE = 100


def main():
    parser = argparse.ArgumentParser(description="Load JSONL chunks into ScholarGraph")
    parser.add_argument("file", help="Path to the JSONL file")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Chunks per API call (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--api-base", default=API_BASE, help="API base URL")
    args = parser.parse_args()

    # Count total lines for progress reporting
    print(f"Counting lines in {args.file}...")
    with open(args.file, "r") as f:
        total = sum(1 for _ in f)
    print(f"Found {total:,} chunks. Loading in batches of {args.batch_size}...")

    client = httpx.Client(timeout=300.0)
    inserted = 0
    batch = []
    start_time = time.time()

    with open(args.file, "r") as f:
        for line in f:
            record = json.loads(line)
            batch.append(record)

            if len(batch) >= args.batch_size:
                resp = client.post(f"{args.api_base}/chunks", json={"chunks": batch})
                resp.raise_for_status()
                inserted += resp.json()["inserted"]
                elapsed = time.time() - start_time
                rate = inserted / elapsed
                eta = (total - inserted) / rate if rate > 0 else 0
                print(f"  {inserted:,}/{total:,} chunks "
                      f"({inserted*100/total:.1f}%) "
                      f"| {rate:.0f} chunks/s "
                      f"| ETA {eta:.0f}s")
                batch = []

    # Final partial batch
    if batch:
        resp = client.post(f"{args.api_base}/chunks", json={"chunks": batch})
        resp.raise_for_status()
        inserted += resp.json()["inserted"]

    elapsed = time.time() - start_time
    print(f"\nDone! Inserted {inserted:,} chunks in {elapsed:.1f}s "
          f"({inserted/elapsed:.0f} chunks/s)")


if __name__ == "__main__":
    main()
