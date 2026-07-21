"""
Retrieval indices for the Credit Risk & AML Advisor.

Loads data/processed/chunks.jsonl (produced by ingestion/ingest.py) and builds:
  - a minsearch.Index for keyword (TF-IDF) search
  - a minsearch.VectorSearch for semantic (embedding) search

Indices and the embedding model are built lazily on first use and cached for
the lifetime of the process, so repeated calls from an eval script or the app
don't rebuild anything twice.

Chunk embeddings are persisted to data/processed/embeddings.npy so that
re-running eval/app processes doesn't re-encode the whole corpus with
sentence-transformers every time (model load + full-corpus encode is the slow
part; encoding a single query at search time is cheap).
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from minsearch import Index, VectorSearch
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"
EMBEDDINGS_FILE = PROCESSED_DIR / "embeddings.npy"
EMBEDDINGS_META_FILE = PROCESSED_DIR / "embeddings.meta.json"

# multi-qa-MiniLM-L6-cos-v1: trained specifically for query<->passage cosine
# similarity (QA-style retrieval), 384-dim, small/fast on CPU. Good fit for
# "question about a regulation" -> "passage from that regulation" matching.
EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"

# Cross-encoder for reranking: scores (query, passage) pairs jointly, which is
# more accurate than bi-encoder cosine similarity but too slow to run over the
# whole corpus — so it re-orders a small candidate pool after retrieval.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TEXT_FIELDS = ["text"]
KEYWORD_FIELDS = ["source_id", "category"]


@lru_cache(maxsize=1)
def get_chunks() -> list[dict]:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"{CHUNKS_FILE} not found. Run `python ingestion/ingest.py` first "
            "to build the knowledge base."
        )
    with open(CHUNKS_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    print(f"[load] embedding model: {EMBEDDING_MODEL} (first call only)")
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    print(f"[load] reranker model: {RERANKER_MODEL} (first call only)")
    return CrossEncoder(RERANKER_MODEL)


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_or_load_embeddings(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    """Encode chunk texts, or load a cached .npy if chunks.jsonl hasn't changed."""
    current_hash = _hash_file(CHUNKS_FILE)

    if EMBEDDINGS_FILE.exists() and EMBEDDINGS_META_FILE.exists():
        meta = json.loads(EMBEDDINGS_META_FILE.read_text())
        if (
            meta.get("source_hash") == current_hash
            and meta.get("model_name") == EMBEDDING_MODEL
            and meta.get("count") == len(chunks)
        ):
            print(f"[cached] loading embeddings from {EMBEDDINGS_FILE.name}")
            return np.load(EMBEDDINGS_FILE)

    print(f"[embed] encoding {len(chunks)} chunks with {EMBEDDING_MODEL} ...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    np.save(EMBEDDINGS_FILE, embeddings)
    EMBEDDINGS_META_FILE.write_text(json.dumps({
        "model_name": EMBEDDING_MODEL,
        "source_hash": current_hash,
        "count": len(chunks),
        "dim": int(embeddings.shape[1]),
    }))
    print(f"[saved] {EMBEDDINGS_FILE.name} {embeddings.shape}")
    return embeddings


@lru_cache(maxsize=1)
def get_keyword_index() -> Index:
    chunks = get_chunks()
    print(f"[index] building keyword index over {len(chunks)} chunks")
    return Index(text_fields=TEXT_FIELDS, keyword_fields=KEYWORD_FIELDS).fit(chunks)


@lru_cache(maxsize=1)
def get_vector_index() -> VectorSearch:
    chunks = get_chunks()
    embeddings = build_or_load_embeddings(chunks, get_embedding_model())
    print(f"[index] building vector index over {len(chunks)} chunks")
    return VectorSearch(keyword_fields=KEYWORD_FIELDS).fit(embeddings, chunks)
