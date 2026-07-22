"""
Cold-start bootstrap for hosts that don't run the Docker build.

The Docker image bakes in the knowledge base and reads the API key from an
env file, so `docker compose up` needs none of this. But on a plain Python
host — Streamlit Community Cloud in particular — a fresh container has:
  - no OPENAI_API_KEY env var (the key lives in the platform's Secrets), and
  - no data/processed/chunks.jsonl (it's gitignored; the repo never ships it).

bootstrap() closes both gaps before the app answers its first question:
expose the secret as an env var (so rag.llm's OpenAI() finds it, exactly as a
local .env would), then build the knowledge base if it isn't there yet.
Idempotent and cheap once the base exists, so it's safe to call on every rerun.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = ROOT / "data" / "processed" / "chunks.jsonl"

# Keys the app reads from the environment; mirror them out of st.secrets when a
# platform provides them there instead of as env vars.
_SECRET_KEYS = ("OPENAI_API_KEY", "LLM_MODEL")


def load_secrets_into_env() -> None:
    import os

    try:
        import streamlit as st

        for key in _SECRET_KEYS:
            if key in st.secrets and not os.environ.get(key):
                os.environ[key] = str(st.secrets[key])
    except Exception:  # noqa: BLE001 - no secrets configured is fine (local/Docker)
        pass


def knowledge_base_ready() -> bool:
    return CHUNKS_FILE.is_file() and CHUNKS_FILE.stat().st_size > 0


def build_knowledge_base() -> None:
    """Download the source PDF and write chunks.jsonl (the app only needs the
    chunks; the minimal ingest.py script avoids the DuckDB write)."""
    from ingestion.ingest import main as ingest_main

    if ingest_main() != 0:
        raise RuntimeError(
            "Ingestion failed to build the knowledge base — check that the host "
            "can reach the source URLs in ingestion/sources.yaml."
        )


def bootstrap() -> None:
    load_secrets_into_env()
    if not knowledge_base_ready():
        build_knowledge_base()
