"""Shared typed data contracts for retrieval, generation, and monitoring."""

from __future__ import annotations

from typing import TypedDict


class Chunk(TypedDict):
    """One auditable passage from the ingested Basel III corpus."""

    chunk_id: str
    source_id: str
    source_title: str
    category: str
    page: int
    text: str


class AnswerResult(TypedDict):
    """Output of the end-to-end RAG answer flow."""

    answer: str
    sources: list[Chunk]
    search_mode: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time: float
    cost: float


class RewriteMetrics(TypedDict):
    """Query-rewrite output and its measured API overhead."""

    query: str
    tokens: int
    cost: float
    time: float
