from rag.search import (
    SEARCH_MODES,
    hybrid_search,
    keyword_search,
    rerank_search,
    vector_search,
)


def answer(*args, **kwargs):
    """Import the LLM layer only when an answer is requested."""
    from rag.llm import answer as generate_answer

    return generate_answer(*args, **kwargs)


__all__ = [
    "keyword_search",
    "vector_search",
    "hybrid_search",
    "rerank_search",
    "SEARCH_MODES",
    "answer",
]
