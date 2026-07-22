"""
Ground-truth dataset generation for retrieval evaluation.

Follows the course's A->Q setup: for a sample of knowledge-base chunks, an
LLM generates questions that each chunk answers. The chunk is then the known
"correct" retrieval target for its questions, which lets us measure hit rate
and MRR without manual labelling.

Run:
    python -m eval.ground_truth

Output:
    eval/ground_truth.csv   (question, chunk_id, source_id, page)

Generation is seeded and sampled (not all 707 chunks) to keep API cost and
runtime small; the sample is large enough for stable mode-level comparisons.
"""

from __future__ import annotations

import csv
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

CHUNKS_FILE = ROOT / "data" / "processed" / "chunks.jsonl"
OUTPUT_FILE = ROOT / "eval" / "ground_truth.csv"

MODEL = "gpt-4o-mini"
SAMPLE_SIZE = 150
QUESTIONS_PER_CHUNK = 3
MIN_CHUNK_CHARS = 300  # skip page stubs that can't support a real question
SEED = 42
MAX_WORKERS = 6


class Questions(BaseModel):
    questions: list[str]


INSTRUCTIONS = """You emulate a credit-risk analyst using our regulatory assistant.
You are given an excerpt from the Basel III standardised approach for credit risk.
Formulate 3 questions this analyst might ask that THIS excerpt answers.
The questions should be complete and specific, not too short, and should reuse
as few words as possible from the excerpt itself. Write them the way a
practitioner would type them into a search box.""".strip()


def load_chunks() -> list[dict]:
    with open(CHUNKS_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]


def generate_for_chunk(client: OpenAI, chunk: dict, max_retries: int = 3) -> list[dict]:
    user_prompt = json.dumps({"page": chunk["page"], "excerpt": chunk["text"]})
    for attempt in range(max_retries):
        try:
            response = client.responses.parse(
                model=MODEL,
                input=[
                    {"role": "developer", "content": INSTRUCTIONS},
                    {"role": "user", "content": user_prompt},
                ],
                text_format=Questions,
                store=False,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise RuntimeError("OpenAI response did not include generated questions")
            return [
                {
                    "question": q,
                    "chunk_id": chunk["chunk_id"],
                    "source_id": chunk["source_id"],
                    "page": chunk["page"],
                }
                for q in parsed.questions[:QUESTIONS_PER_CHUNK]
            ]
        except Exception:  # noqa: BLE001
            if attempt == max_retries - 1:
                raise
            time.sleep(2**attempt)
    return []


def main() -> int:
    chunks = load_chunks()
    eligible = [c for c in chunks if len(c["text"]) >= MIN_CHUNK_CHARS]
    random.seed(SEED)
    sample = random.sample(eligible, min(SAMPLE_SIZE, len(eligible)))
    print(f"Generating {QUESTIONS_PER_CHUNK} questions for {len(sample)} of "
          f"{len(chunks)} chunks (model: {MODEL}, seed: {SEED})")

    client = OpenAI()
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for result in tqdm(
            pool.map(lambda c: generate_for_chunk(client, c), sample),
            total=len(sample),
        ):
            rows.extend(result)

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "chunk_id", "source_id", "page"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} ground-truth rows -> {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
