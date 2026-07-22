"""
dlt-based ingestion pipeline for the Credit Risk Advisor.

Same flow as ingest.py (download PDF -> extract -> clean -> chunk) but the
chunk records are loaded with dlt into a DuckDB knowledge base, with
chunks.jsonl written from the same data for the retrieval layer.

Run:
    python ingestion/dlt_pipeline.py

Output:
    data/processed/kb.duckdb      (dlt-managed DuckDB knowledge base)
    data/processed/chunks.jsonl   (consumed by rag/index.py)

The download/extract/chunk logic lives in ingest.py and is reused here —
this file only adds the dlt load. ingest.py also still works standalone as
a minimal-dependency fallback.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict

import dlt

from ingest import PROCESSED_DIR, download, extract_and_chunk, load_sources

KB_FILE = PROCESSED_DIR / "kb.duckdb"
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"


def build_chunks() -> tuple[list[dict], int]:
    """Download all sources and produce chunk dicts. Returns (chunks, failures)."""
    all_chunks: list[dict] = []
    failed = 0
    for source in load_sources():
        print(f"Processing: {source['id']}")
        pdf_path = download(source)
        if pdf_path is None:
            failed += 1
            continue
        chunks = extract_and_chunk(pdf_path, source)
        print(f"  [chunked] {len(chunks)} chunks\n")
        all_chunks.extend(asdict(c) for c in chunks)
    return all_chunks, failed


def main() -> int:
    chunk_dicts, failed = build_chunks()

    if not chunk_dicts:
        print("No chunks produced. Check network access to the source URLs "
              "(or place the PDFs manually in data/raw/ — see README).")
        return 1

    @dlt.resource(name="chunks", write_disposition="replace", primary_key="chunk_id")
    def chunk_resource():
        yield from chunk_dicts

    pipeline = dlt.pipeline(
        pipeline_name="credit_risk_kb",
        destination=dlt.destinations.duckdb(str(KB_FILE)),
        dataset_name="knowledge_base",
    )
    load_info = pipeline.run(chunk_resource())
    print(load_info)

    with open(CHUNKS_FILE, "w") as f:
        for c in chunk_dicts:
            f.write(json.dumps(c) + "\n")
    print(f"Wrote {len(chunk_dicts)} chunks -> {CHUNKS_FILE}")

    if failed:
        print(f"ERROR: {failed} source(s) failed to download — the knowledge "
              "base is incomplete.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
