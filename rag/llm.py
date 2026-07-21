"""
LLM answer generation for the Credit Risk & AML Advisor.

Takes a question, retrieves grounding context via rag.search, and calls the
OpenAI Responses API with system-level rules in the `developer` role and the
question+context as the user prompt (the same split the course's RAGBase
uses). The model is instructed to answer only from the retrieved context,
with citations.

Requires OPENAI_API_KEY in a .env file at the repo root (see .env.example).

Run as a manual smoke test:
    python -m rag.llm
    python -m rag.llm "what are the red flags for structuring?"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from rag.search import DEFAULT_NUM_RESULTS, SEARCH_MODES

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

# rerank won the retrieval evaluation (python -m eval.evaluate_retrieval):
# hit rate 0.911 / MRR 0.768 vs 0.816/0.595 for the next-best mode (hybrid),
# at ~184ms — acceptable for an interactive assistant. See README Evaluation.
DEFAULT_SEARCH_MODE = "rerank"

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()  # reads OPENAI_API_KEY from the environment
    return _client


_COMMON_RULES = """Rules:
- Base your answer strictly on the CONTEXT. Do not use outside knowledge.
- If the CONTEXT does not contain enough information to answer, say so
  plainly instead of guessing.
- Cite the source and page for each claim, using the format
  [source_title, p.page] matching the CONTEXT entries.
- This is an educational tool, not legal or compliance advice; do not phrase
  the answer as a directive to take a specific action."""

INSTRUCTIONS_CITED = f"""You are a compliance and credit-risk research assistant. Answer the
QUESTION using ONLY the CONTEXT provided, which consists of excerpts from
public banking regulation and supervisory guidance documents.

{_COMMON_RULES}""".strip()

INSTRUCTIONS_QUOTE_FIRST = f"""You are a compliance and credit-risk research assistant. Answer the
QUESTION using ONLY the CONTEXT provided, which consists of excerpts from
public banking regulation and supervisory guidance documents.

Method:
1. First identify the specific clause(s) in the CONTEXT that govern the
   question, and quote the decisive phrase briefly.
2. Then state the answer in one or two sentences.

{_COMMON_RULES}""".strip()

# quote-first won the LLM evaluation (python -m eval.evaluate_llm):
# good_rate 0.820 vs 0.805 for the plain cited prompt (LLM-as-judge,
# 200 questions). Margin is small; see README Evaluation for caveats.
INSTRUCTIONS = INSTRUCTIONS_QUOTE_FIRST

USER_PROMPT_TEMPLATE = """QUESTION: {question}

CONTEXT:
{context}""".strip()


def build_context(chunks: list[dict]) -> str:
    entries = [f"[{c['source_title']}, p.{c['page']}]\n{c['text']}" for c in chunks]
    return "\n\n---\n\n".join(entries)


def build_prompt(question: str, chunks: list[dict]) -> str:
    return USER_PROMPT_TEMPLATE.format(question=question, context=build_context(chunks))


def llm(user_prompt: str, instructions: str = INSTRUCTIONS) -> str:
    response = get_client().responses.create(
        model=LLM_MODEL,
        input=[
            {"role": "developer", "content": instructions},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    return response.output_text


def answer(
    question: str,
    search_mode: str = DEFAULT_SEARCH_MODE,
    num_results: int = DEFAULT_NUM_RESULTS,
) -> dict:
    """Run the full RAG flow: retrieve -> build prompt -> call the LLM.

    Returns {"answer", "sources", "search_mode"} so the app layer can show
    citations alongside the generated answer.
    """
    search_fn = SEARCH_MODES[search_mode]
    chunks = search_fn(question, num_results=num_results)
    if not chunks:
        return {
            "answer": "I couldn't find anything in the knowledge base relevant to this question.",
            "sources": [],
            "search_mode": search_mode,
        }
    prompt = build_prompt(question, chunks)
    generated = llm(prompt)
    return {"answer": generated, "sources": chunks, "search_mode": search_mode}


DEMO_QUESTIONS = [
    "What is the risk weight for a corporate exposure with no external credit rating under the standardised approach?",
    # Off-corpus question: demonstrates the grounded refusal instead of hallucination.
    "What is the capital requirement for operational risk?",
]


def main() -> int:
    questions = sys.argv[1:] or DEMO_QUESTIONS
    for question in questions:
        print(f"\nQ: {question}")
        result = answer(question)
        print(f"\nA: {result['answer']}")
        print("\nSources:")
        for c in result["sources"]:
            print(f"  - {c['source_title']}, p.{c['page']} ({c['chunk_id']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
