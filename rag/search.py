"""
Retrieval entry point for the Credit Risk Advisor.

Exposes four search modes over data/processed/chunks.jsonl:
  - keyword_search  : TF-IDF / exact-term matching (minsearch.Index)
  - vector_search   : sentence-transformer embeddings (minsearch.VectorSearch)
  - hybrid_search   : reciprocal rank fusion of the two above
  - rerank_search   : hybrid candidates re-ordered by a cross-encoder

All three return a list of chunk dicts (chunk_id, source_id, source_title,
category, page, text), ranked best-first, so downstream code (retrieval
eval, prompt building) doesn't need to know which mode produced them.

minsearch's search() methods don't expose a relevance score for either
mode (only ranked payload dicts), so hybrid combines them by rank
(reciprocal rank fusion) rather than blending scores that aren't comparable
across TF-IDF cosine and embedding cosine anyway.

Run as a manual smoke test:
    python -m rag.search
    python -m rag.search "what are red flags for structuring?"
"""

from __future__ import annotations

import sys

from rag.index import (
    get_embedding_model,
    get_keyword_index,
    get_reranker,
    get_vector_index,
)

DEFAULT_NUM_RESULTS = 5
# Tuned on eval/ground_truth.csv (python -m eval.evaluate_retrieval --tune):
# k=10 beat the classic k=60 default on hit rate (0.822 vs 0.796); candidate
# pool size had little effect, so the 4x formula below is kept.
RRF_K = 10


def keyword_search(
    query: str, num_results: int = DEFAULT_NUM_RESULTS, filter_dict: dict | None = None
) -> list[dict]:
    return get_keyword_index().search(query, filter_dict=filter_dict, num_results=num_results)


def vector_search(
    query: str, num_results: int = DEFAULT_NUM_RESULTS, filter_dict: dict | None = None
) -> list[dict]:
    query_vec = get_embedding_model().encode(query, normalize_embeddings=True)
    return get_vector_index().search(query_vec, filter_dict=filter_dict, num_results=num_results)


def _reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = RRF_K) -> list[dict]:
    scores: dict[str, float] = {}
    doc_by_id: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            cid = doc["chunk_id"]
            doc_by_id[cid] = doc
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    ordered_ids = sorted(scores, key=scores.get, reverse=True)
    return [doc_by_id[cid] for cid in ordered_ids]


def hybrid_search(
    query: str,
    num_results: int = DEFAULT_NUM_RESULTS,
    filter_dict: dict | None = None,
    rrf_k: int = RRF_K,
    num_candidates: int | None = None,
) -> list[dict]:
    if num_candidates is None:
        num_candidates = max(num_results * 4, 20)  # wider pool gives fusion overlap to work with
    kw_results = keyword_search(query, num_results=num_candidates, filter_dict=filter_dict)
    vec_results = vector_search(query, num_results=num_candidates, filter_dict=filter_dict)
    fused = _reciprocal_rank_fusion([kw_results, vec_results], k=rrf_k)
    return fused[:num_results]


def rerank_search(
    query: str, num_results: int = DEFAULT_NUM_RESULTS, filter_dict: dict | None = None
) -> list[dict]:
    """Hybrid retrieval followed by cross-encoder reranking.

    The cross-encoder scores each (query, chunk_text) pair jointly — a
    distinct, stronger relevance signal than the rank fusion hybrid_search
    already does — but is too slow for the full corpus, so it only re-orders
    a candidate pool.
    """
    num_candidates = max(num_results * 4, 20)
    candidates = hybrid_search(query, num_results=num_candidates, filter_dict=filter_dict)
    if not candidates:
        return []
    scores = get_reranker().predict([(query, c["text"]) for c in candidates])
    ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
    return [c for _, c in ranked[:num_results]]


# Registry the eval script (and later the app) can iterate over uniformly
# without special-casing each mode.
SEARCH_MODES = {
    "keyword": keyword_search,
    "vector": vector_search,
    "hybrid": hybrid_search,
    "rerank": rerank_search,
}

DEMO_QUERIES = [
    "What is the risk weight for a corporate exposure with no external credit rating under the standardised approach?",
    "What collateral is eligible for credit risk mitigation?",
    "How are exposures to small and medium-sized enterprises treated?",
]


def _print_results(mode: str, results: list[dict]) -> None:
    print(f"  [{mode}] ({len(results)} results)")
    for rank, r in enumerate(results, start=1):
        snippet = r["text"][:160].replace("\n", " ")
        print(f"    {rank}. {r['chunk_id']}  (p.{r['page']}, {r['source_id']}) - {snippet}...")


def main() -> int:
    queries = sys.argv[1:] or DEMO_QUERIES
    for query in queries:
        print(f"\nQuery: {query!r}")
        for mode_name, search_fn in SEARCH_MODES.items():
            _print_results(mode_name, search_fn(query, num_results=3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
