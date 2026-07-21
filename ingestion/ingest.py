"""
Automated ingestion pipeline for the Credit Risk & AML Advisor.

Flow:  download PDF  ->  extract text  ->  clean  ->  chunk  ->  write JSONL

Run:
    python ingestion/ingest.py

Output:
    data/processed/chunks.jsonl   (one JSON object per chunk)

This is deliberately a plain Python script (not a notebook) so that ingestion
is automated and reproducible — a single command rebuilds the knowledge base
from scratch on any machine with internet access.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import requests
import yaml
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SOURCES_FILE = ROOT / "ingestion" / "sources.yaml"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Chunking parameters. ~1000 characters with 150 overlap keeps chunks small
# enough to be precise for retrieval but large enough to hold a full clause.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    source_title: str
    category: str
    page: int
    text: str


def load_sources() -> list[dict]:
    with open(SOURCES_FILE) as f:
        return yaml.safe_load(f)["sources"]


def download(source: dict) -> Path | None:
    """Download a source PDF into data/raw. Skips if already present."""
    dest = RAW_DIR / f"{source['id']}.pdf"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [cached] {dest.name}")
        return dest

    print(f"  [download] {source['url']}")
    try:
        resp = requests.get(source["url"], timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001
        print(f"  [ERROR] could not download {source['id']}: {e}")
        return None

    # Validate it's actually a PDF, not an HTML error page.
    if not resp.content[:5].startswith(b"%PDF"):
        print(f"  [ERROR] {source['id']} did not return a PDF (got HTML?). "
              f"Check the URL in sources.yaml.")
        return None

    dest.write_bytes(resp.content)
    print(f"  [saved] {dest.name} ({len(resp.content) // 1024} KB)")
    return dest


def clean_text(text: str) -> str:
    """Normalise whitespace and strip obvious PDF artefacts."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sliding-window character chunking with overlap.

    Overlap matters: it prevents a relevant sentence from being split across a
    boundary and lost to retrieval. See etc/chunking notes in the README.
    """
    if len(text) <= size:
        return [text] if text.strip() else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def extract_and_chunk(pdf_path: Path, source: dict) -> list[Chunk]:
    reader = PdfReader(str(pdf_path))
    chunks: list[Chunk] = []
    for page_num, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        cleaned = clean_text(raw)
        for i, piece in enumerate(chunk_text(cleaned)):
            chunks.append(
                Chunk(
                    chunk_id=f"{source['id']}_p{page_num}_c{i}",
                    source_id=source["id"],
                    source_title=source["title"],
                    category=source["category"],
                    page=page_num,
                    text=piece,
                )
            )
    return chunks


def main() -> int:
    sources = load_sources()
    print(f"Loaded {len(sources)} source(s) from {SOURCES_FILE.name}\n")

    all_chunks: list[Chunk] = []
    failed = 0
    for source in sources:
        print(f"Processing: {source['id']}")
        pdf_path = download(source)
        if pdf_path is None:
            failed += 1
            continue
        chunks = extract_and_chunk(pdf_path, source)
        print(f"  [chunked] {len(chunks)} chunks\n")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No chunks produced. Check network access to the source URLs.")
        print("(These regulator domains may be blocked in some sandboxes — "
              "run this on a normal machine with internet.)")
        return 1

    out_file = PROCESSED_DIR / "chunks.jsonl"
    with open(out_file, "w") as f:
        for c in all_chunks:
            f.write(json.dumps(asdict(c)) + "\n")

    print(f"Wrote {len(all_chunks)} chunks -> {out_file}")
    if failed:
        print(f"ERROR: {failed} source(s) failed to download — the knowledge "
              "base is incomplete.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
