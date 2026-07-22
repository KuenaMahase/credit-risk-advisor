"""
Query-rewriting evaluation: does LLM query rewriting improve retrieval?

Compares the production retrieval mode (rerank) on the ground-truth questions
as-is versus after rag.llm.rewrite_query() rewrites them into Basel
terminology. Reports hit rate / MRR plus what rewriting costs per query
(latency, tokens, dollars), so the decision to enable it is evidence-based.

The per-question rewrites are persisted to eval/rewrite_queries.csv (committed)
and reused on the next run, so a reviewer can reproduce the comparison and
audit every rewrite without making 450 fresh API calls. Delete that file (or
pass --refresh) to regenerate the rewrites from the model.

Caveat documented with the results: the ground-truth questions are themselves
LLM-generated from framework text, so they already use framework vocabulary —
this measures rewriting's effect on well-formed queries, its best case being
"no harm". Terse practitioner shorthand (the motivating case) is not in the
ground truth.

Run:
    python -m eval.evaluate_rewrite            # uses the cached rewrites if present
    python -m eval.evaluate_rewrite --refresh  # regenerates rewrites (450 API calls)

Output:
    eval/rewrite_queries.csv   (per-question rewrites — the audit trail / cache)
    eval/rewrite_results.csv   (aggregate comparison)
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
REWRITES_FILE = ROOT / "eval" / "rewrite_queries.csv"
RESULTS_FILE = ROOT / "eval" / "rewrite_results.csv"

REWRITE_FIELDS = ["question", "chunk_id", "rewritten_query",
                  "input_tokens", "output_tokens", "latency_ms"]

SEARCH_MODE = "rerank"
MAX_WORKERS = 6
NUM_RESULTS = 5


def rewrite_one(q: dict) -> dict:
    """Rewrite a single question. Returns a self-contained per-question row —
    no shared mutable state, so this is safe to run across threads."""
    start = time.perf_counter()
    rewritten, usage = rewrite_query_with_usage(q["question"])
    return {
        "question": q["question"],
        "chunk_id": q["chunk_id"],
        "rewritten_query": rewritten,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "latency_ms": 1000 * (time.perf_counter() - start),
    }


def compute_rewrites(ground_truth: list[dict]) -> list[dict]:
    """Rewrite all questions in parallel. pool.map preserves input order, and
    each row is built independently, so aggregation afterwards is race-free."""
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        return list(tqdm(pool.map(rewrite_one, ground_truth),
                         total=len(ground_truth), desc="rewriting"))


def load_cached_rewrites(ground_truth: list[dict]) -> list[dict] | None:
    """Return cached per-question rewrite rows aligned to ground_truth, or None
    if the cache is missing or doesn't cover every (question, chunk_id)."""
    if not REWRITES_FILE.exists():
        return None
    with open(REWRITES_FILE, newline="") as f:
        cached = {(r["question"], r["chunk_id"]): r for r in csv.DictReader(f)}
    rows = []
    for q in ground_truth:
        row = cached.get((q["question"], q["chunk_id"]))
        if row is None:
            return None
        rows.append({
            **row,
            "input_tokens": int(row["input_tokens"]),
            "output_tokens": int(row["output_tokens"]),
            "latency_ms": float(row["latency_ms"]),
        })
    return rows


def save_rewrites(rows: list[dict]) -> None:
    with open(REWRITES_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REWRITE_FIELDS)
        writer.writeheader()
        writer.writerows({k: r[k] for k in REWRITE_FIELDS} for r in rows)


def rewrite_stats(rows: list[dict]) -> dict:
    n = len(rows)
    input_tokens = sum(r["input_tokens"] for r in rows)
    output_tokens = sum(r["output_tokens"] for r in rows)
    return {
        "avg_rewrite_latency_ms": sum(r["latency_ms"] for r in rows) / n,
        "avg_tokens_per_rewrite": (input_tokens + output_tokens) / n,
        "total_cost_usd": (input_tokens * PRICE_PER_M_INPUT
                           + output_tokens * PRICE_PER_M_OUTPUT) / 1_000_000,
    }


def evaluate_queries(ground_truth: list[dict], queries: list[str], desc: str) -> dict:
    search_fn = SEARCH_MODES[SEARCH_MODE]
    relevance = []
    for q, query in tqdm(list(zip(ground_truth, queries)), desc=desc, leave=False):
        results = search_fn(query, num_results=NUM_RESULTS)
        relevance.append([int(d["chunk_id"] == q["chunk_id"]) for d in results])
    return {"hit_rate": hit_rate(relevance), "mrr": mrr(relevance)}


def main() -> int:
    refresh = "--refresh" in sys.argv
    ground_truth = load_ground_truth()
    print(f"Comparing original vs rewritten queries on {len(ground_truth)} "
          f"ground-truth questions (mode: {SEARCH_MODE}, top-{NUM_RESULTS})\n")

    # Warm the retrieval stack serially before any thread pool touches it.
    SEARCH_MODES[SEARCH_MODE]("warm-up query")

    original = evaluate_queries(ground_truth, [q["question"] for q in ground_truth], "original")
    print(f"original : hit_rate={original['hit_rate']:.3f}  mrr={original['mrr']:.3f}")

    cached = None if refresh else load_cached_rewrites(ground_truth)
    if cached is not None:
        print(f"[cached] loaded rewrites from {REWRITES_FILE.name} (no API calls)")
        rewrite_rows = cached
    else:
        rewrite_rows = compute_rewrites(ground_truth)
        save_rewrites(rewrite_rows)
        print(f"[saved] rewrites -> {REWRITES_FILE.name}")

    stats = rewrite_stats(rewrite_rows)
    rewritten = evaluate_queries(
        ground_truth, [r["rewritten_query"] for r in rewrite_rows], "rewritten")
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
