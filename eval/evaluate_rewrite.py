"""
Query-rewriting evaluation: does LLM query rewriting improve retrieval?

Compares the production retrieval mode (rerank) on the ground-truth questions
as-is versus after rag.llm.rewrite_query() rewrites them into Basel
terminology. Reports hit rate / MRR plus what rewriting costs per query
(latency, tokens, dollars), so the decision to enable it is evidence-based.

Caveat documented with the results: the ground-truth questions are themselves
LLM-generated from framework text, so they already use framework vocabulary —
this measures rewriting's effect on well-formed queries, its best case being
"no harm". Terse practitioner shorthand (the motivating case) is not in the
ground truth.

Run:
    python -m eval.evaluate_rewrite

Output:
    eval/rewrite_results.csv
"""

from __future__ import annotations

import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from eval.evaluate_retrieval import hit_rate, load_ground_truth, mrr
from rag.llm import PRICE_PER_M_INPUT, PRICE_PER_M_OUTPUT, rewrite_query_with_usage
from rag.search import SEARCH_MODES

ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "eval" / "rewrite_results.csv"

SEARCH_MODE = "rerank"
MAX_WORKERS = 6
NUM_RESULTS = 5


def rewrite_all(ground_truth: list[dict]) -> tuple[list[str], dict]:
    """Rewrite every ground-truth question in parallel; return texts + stats."""
    latencies: list[float] = []
    input_tokens = 0
    output_tokens = 0

    def rewrite_one(q: dict) -> str:
        start = time.perf_counter()
        rewritten, usage = rewrite_query_with_usage(q["question"])
        latencies.append(time.perf_counter() - start)
        nonlocal input_tokens, output_tokens
        input_tokens += usage.input_tokens
        output_tokens += usage.output_tokens
        return rewritten

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        rewritten = list(tqdm(pool.map(rewrite_one, ground_truth),
                              total=len(ground_truth), desc="rewriting"))

    n = len(ground_truth)
    stats = {
        "avg_rewrite_latency_ms": 1000 * sum(latencies) / n,
        "avg_tokens_per_rewrite": (input_tokens + output_tokens) / n,
        "total_cost_usd": (input_tokens * PRICE_PER_M_INPUT
                           + output_tokens * PRICE_PER_M_OUTPUT) / 1_000_000,
    }
    return rewritten, stats


def evaluate_queries(ground_truth: list[dict], queries: list[str], desc: str) -> dict:
    search_fn = SEARCH_MODES[SEARCH_MODE]
    relevance = []
    for q, query in tqdm(list(zip(ground_truth, queries)), desc=desc, leave=False):
        results = search_fn(query, num_results=NUM_RESULTS)
        relevance.append([int(d["chunk_id"] == q["chunk_id"]) for d in results])
    return {"hit_rate": hit_rate(relevance), "mrr": mrr(relevance)}


def main() -> int:
    ground_truth = load_ground_truth()
    print(f"Comparing original vs rewritten queries on {len(ground_truth)} "
          f"ground-truth questions (mode: {SEARCH_MODE}, top-{NUM_RESULTS})\n")

    # Warm the retrieval stack serially before any thread pool touches it.
    SEARCH_MODES[SEARCH_MODE]("warm-up query")

    original = evaluate_queries(ground_truth, [q["question"] for q in ground_truth], "original")
    print(f"original : hit_rate={original['hit_rate']:.3f}  mrr={original['mrr']:.3f}")

    rewritten_queries, stats = rewrite_all(ground_truth)
    rewritten = evaluate_queries(ground_truth, rewritten_queries, "rewritten")
    print(f"rewritten: hit_rate={rewritten['hit_rate']:.3f}  mrr={rewritten['mrr']:.3f}")
    print(f"rewrite overhead: {stats['avg_rewrite_latency_ms']:.0f}ms/query, "
          f"{stats['avg_tokens_per_rewrite']:.0f} tokens/query, "
          f"${stats['total_cost_usd']:.4f} total for {len(ground_truth)} queries")

    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "variant", "hit_rate", "mrr",
            "avg_rewrite_latency_ms", "avg_tokens_per_rewrite", "total_cost_usd",
        ])
        writer.writeheader()
        writer.writerow({"variant": "original", **original,
                         "avg_rewrite_latency_ms": 0,
                         "avg_tokens_per_rewrite": 0, "total_cost_usd": 0})
        writer.writerow({"variant": "rewritten", **rewritten, **stats})
    print(f"\nWrote results -> {RESULTS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
