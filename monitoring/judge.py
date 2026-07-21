"""
Online LLM-as-a-judge for the Credit Risk Advisor app.

Unlike eval/evaluate_llm.py (offline, needs ground truth), this judge runs
on live traffic with no ground truth: it classifies how relevant a generated
answer is to the question, so every conversation gets an automatic quality
signal alongside any user thumbs feedback. Verdicts are stored in the
feedback table with source='judge'.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel

from rag.llm import LLM_MODEL, get_client


class RelevanceVerdict(BaseModel):
    relevance: Literal["NON_RELEVANT", "PARTLY_RELEVANT", "RELEVANT"]
    explanation: str


JUDGE_INSTRUCTIONS = """You are an expert evaluator for a RAG system over the Basel III
credit-risk framework. Analyze the relevance of the generated answer to the
given question.

Classify the answer as:
- RELEVANT: the answer addresses the question
- PARTLY_RELEVANT: the answer partially addresses the question
- NON_RELEVANT: the answer does not address the question

A grounded refusal ("the context does not contain...") counts as RELEVANT
only if it correctly tells the user the knowledge base can't answer;
otherwise judge the substance of the answer.""".strip()

JUDGE_PROMPT = """Question: {question}
Generated Answer: {answer}""".strip()


def evaluate_relevance(question: str, answer: str, max_retries: int = 3) -> tuple[str, str]:
    """Return (relevance, explanation) for a live question/answer pair."""
    prompt = JUDGE_PROMPT.format(question=question, answer=answer)
    for attempt in range(max_retries):
        try:
            response = get_client().responses.parse(
                model=LLM_MODEL,
                input=[
                    {"role": "developer", "content": JUDGE_INSTRUCTIONS},
                    {"role": "user", "content": prompt},
                ],
                text_format=RelevanceVerdict,
            )
            verdict = response.output_parsed
            return verdict.relevance, verdict.explanation
        except Exception:  # noqa: BLE001
            if attempt == max_retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")
