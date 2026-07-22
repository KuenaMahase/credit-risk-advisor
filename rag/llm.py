"""
LLM answer generation for the Credit Risk Advisor.

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
import time
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses.response_usage import ResponseUsage

from rag.search import DEFAULT_NUM_RESULTS, SEARCH_MODES
from rag.types import AnswerResult, Chunk

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

# Standard text-token prices in USD per 1M tokens. Custom models can be used by
# setting both LLM_PRICE_PER_M_INPUT and LLM_PRICE_PER_M_OUTPUT in the
# environment; requiring an explicit pair prevents silently incorrect costs.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
}


class TokenUsage(Protocol):
    """Token fields required for cost calculation."""

    input_tokens: int
    output_tokens: int


def get_model_pricing(model: str = LLM_MODEL) -> tuple[float, float]:
    input_override = os.environ.get("LLM_PRICE_PER_M_INPUT")
    output_override = os.environ.get("LLM_PRICE_PER_M_OUTPUT")
    if bool(input_override) != bool(output_override):
        raise ValueError(
            "Set both LLM_PRICE_PER_M_INPUT and LLM_PRICE_PER_M_OUTPUT, or neither."
        )
    if input_override and output_override:
        return float(input_override), float(output_override)
    try:
        return MODEL_PRICING[model]
    except KeyError as exc:
        raise ValueError(
            f"No pricing configured for {model!r}. Set LLM_PRICE_PER_M_INPUT "
            "and LLM_PRICE_PER_M_OUTPUT."
        ) from exc


PRICE_PER_M_INPUT, PRICE_PER_M_OUTPUT = get_model_pricing()


def calculate_cost(usage: TokenUsage, model: str = LLM_MODEL) -> float:
    """Return the standard API token cost for a response usage object."""
    input_price, output_price = get_model_pricing(model)
    return (
        usage.input_tokens * input_price + usage.output_tokens * output_price
    ) / 1_000_000


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
- Keep direct quotations brief and below 100 words in total.
- Do not imply affiliation with or endorsement by the source publisher.
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

REWRITE_INSTRUCTIONS = """You rewrite an analyst's search query about the Basel III standardised
approach for credit risk so that it retrieves the right passage from the
framework text.

- Expand abbreviations and informal phrasing into the framework's own
  terminology, e.g. CCF -> credit conversion factor, LTV -> loan-to-value
  ratio, CRM -> credit risk mitigation, RW -> risk weight, PD/LGD/EAD ->
  probability of default / loss given default / exposure at default,
  QCCP -> qualifying central counterparty, SFT -> securities financing
  transaction, reso mortgage/home loan -> residential real estate exposure.
- Keep it a single concise question; do not add topics the analyst did not ask about.
- Output ONLY the rewritten query, nothing else.""".strip()


def rewrite_query_with_usage(question: str) -> tuple[str, ResponseUsage]:
    """Rewrite a user query into framework terminology. Returns (query, usage)."""
    response = get_client().responses.create(
        model=LLM_MODEL,
        input=[
            {"role": "developer", "content": REWRITE_INSTRUCTIONS},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
        store=False,
    )
    usage = response.usage
    if usage is None:
        raise RuntimeError("OpenAI response did not include token usage")
    return response.output_text.strip(), usage


def rewrite_query(question: str) -> str:
    rewritten, _ = rewrite_query_with_usage(question)
    return rewritten

USER_PROMPT_TEMPLATE = """QUESTION: {question}

CONTEXT:
{context}""".strip()


def build_context(chunks: list[Chunk]) -> str:
    entries = [f"[{c['source_title']}, p.{c['page']}]\n{c['text']}" for c in chunks]
    return "\n\n---\n\n".join(entries)


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    return USER_PROMPT_TEMPLATE.format(question=question, context=build_context(chunks))


def llm_with_usage(
    user_prompt: str, instructions: str = INSTRUCTIONS
) -> tuple[str, ResponseUsage]:
    """Call the LLM and return (answer_text, usage) for metrics logging."""
    response = get_client().responses.create(
        model=LLM_MODEL,
        input=[
            {"role": "developer", "content": instructions},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        store=False,
    )
    usage = response.usage
    if usage is None:
        raise RuntimeError("OpenAI response did not include token usage")
    return response.output_text, usage


def llm(user_prompt: str, instructions: str = INSTRUCTIONS) -> str:
    text, _ = llm_with_usage(user_prompt, instructions)
    return text


def answer(
    question: str,
    search_mode: str = DEFAULT_SEARCH_MODE,
    num_results: int = DEFAULT_NUM_RESULTS,
) -> AnswerResult:
    """Run the full RAG flow: retrieve -> build prompt -> call the LLM.

    Returns the answer plus sources and per-call metrics (tokens, cost,
    response time) so the app layer can show citations and the monitoring
    store can log every conversation.
    """
    start = time.perf_counter()
    search_fn = SEARCH_MODES[search_mode]
    chunks = search_fn(question, num_results=num_results)
    if not chunks:
        return {
            "answer": "I couldn't find anything in the knowledge base relevant to this question.",
            "sources": [],
            "search_mode": search_mode,
            "model": LLM_MODEL,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "response_time": time.perf_counter() - start,
            "cost": 0.0,
        }
    prompt = build_prompt(question, chunks)
    generated, usage = llm_with_usage(prompt)
    cost = calculate_cost(usage)
    return {
        "answer": generated,
        "sources": chunks,
        "search_mode": search_mode,
        "model": LLM_MODEL,
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "response_time": time.perf_counter() - start,
        "cost": cost,
    }


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
