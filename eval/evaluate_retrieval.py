"""
Retrieval evaluation: hit rate + MRR + latency across all search modes.

Uses eval/ground_truth.csv (produced by eval/ground_truth.py): each row is a
question whose known-correct retrieval target is the chunk it was generated
from. A retrieved chunk counts as relevant only if its chunk_id matches
exactly — a strict criterion given the 150-char chunk overlap, but applied
identically to every mode, so the comparison is fair.

Run:
    python -m eval.evaluate_retrieval            # compare all SEARCH_MODES
    python -m eval.evaluate_retrieval --tune     # also grid-search RRF k / pool size

Output:
    eval/retrieval_results.csv   (mode, hit_rate, mrr, avg_latency_ms)
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

from tqdm import tqdm

from rag.search import SEARCH_MODES, hybrid_search

ROOT = Path(__file__).resolve().parent.parent
GROUND_TRUTH_FILE = ROOT / "eval" / "ground_truth.csv"
RESULTS_FILE = ROOT / "eval" / "retrieval_results.csv"

NUM_RESULTS = 5


def load_ground_truth() -> list[dict]:
    with open(GROUND_TRUTH_FILE) as f:
        return list(csv.DictReader(f))


def hit_rate(relevance: list[list[int]]) -> float:
    return sum(1 for row in relevance if 1 in row) / len(relevance)


def mrr(relevance: list[list[int]]) -> float:
    total = 0.0
    for row in relevance:
        for rank, rel in enumerate(row):
            if rel == 1:
                total += 1 / (rank + 1)
                break
    return total / len(relevance)


def evaluate(ground_truth: list[dict], search_fn, desc: str) -> dict:
    relevance: list[list[int]] = []
    latencies: list[float] = []
    for q in tqdm(ground_truth, desc=desc, leave=False):
        start = time.perf_counter()
        results = search_fn(q["question"], num_results=NUM_RESULTS)
        latencies.append(time.perf_counter() - start)
        relevance.append([int(d["chunk_id"] == q["chunk_id"]) for d in results])
    return {
        "hit_rate": hit_rate(relevance),
        "mrr": mrr(relevance),
        "avg_latency_ms": 1000 * sum(latencies) / len(latencies),
    }


def tune_rrf(ground_truth: list[dict]) -> None:
    """Grid-search RRF k and candidate-pool size for hybrid search."""
    print("\nRRF tuning (hybrid):")
    print(f"{'k':>6} {'candidates':>11} {'hit_rate':>9} {'mrr':>7}")
    for k in [10, 60, 120]:
        for num_candidates in [10, 20, 40]:
            def fn(query, num_results=NUM_RESULTS, k=k, nc=num_candidates):
                return hybrid_search(
                    query, num_results=num_results, rrf_k=k, num_candidates=nc
                )
            m = evaluate(ground_truth, fn, desc=f"k={k} nc={num_candidates}")
            print(f"{k:>6} {num_candidates:>11} {m['hit_rate']:>9.3f} {m['mrr']:>7.3f}")


def main() -> int:
    ground_truth = load_ground_truth()
    print(f"Evaluating {len(SEARCH_MODES)} search modes on "
          f"{len(ground_truth)} ground-truth questions (top-{NUM_RESULTS})\n")

    rows = []
    for mode, search_fn in SEARCH_MODES.items():
        metrics = evaluate(ground_truth, search_fn, desc=mode)
        rows.append({"mode": mode, **metrics})
        print(f"{mode:>8}: hit_rate={metrics['hit_rate']:.3f}  "
              f"mrr={metrics['mrr']:.3f}  "
              f"latency={metrics['avg_latency_ms']:.0f}ms")

    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "hit_rate", "mrr", "avg_latency_ms"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote results -> {RESULTS_FILE}")

    if "--tune" in sys.argv:
        tune_rrf(ground_truth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
