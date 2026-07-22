"""
Streamlit UI for the Credit Risk Advisor.

Run from the repo root:
    streamlit run app/app.py

Flow per question: retrieve + generate a cited answer (rag.llm.answer) ->
log the conversation to the monitoring store -> auto-judge relevance
(online LLM judge, saved as feedback source='judge') -> offer the user
thumbs up/down (saved as feedback source='user').
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, TypedDict, cast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from monitoring.db import save_conversation, save_feedback, update_conversation_judge
from monitoring.judge import evaluate_relevance
from rag.llm import (
    DEFAULT_SEARCH_MODE,
    answer,
    calculate_cost,
    rewrite_query_with_usage,
)
from rag.search import SEARCH_MODES
from rag.types import AnswerResult, RewriteMetrics


class LastInteraction(TypedDict):
    """Typed Streamlit session payload for the most recent answer."""

    conversation_id: int
    question: str
    rewrite: RewriteMetrics | None
    result: AnswerResult
    relevance: str | None
    explanation: str | None
    voted: bool


def render_app(set_page_config: bool = True) -> None:
    """Render the assistant.

    The cloud entrypoint configures the page before reading Streamlit secrets,
    then calls this function with ``set_page_config=False``. Local and Docker
    runs execute this module directly and keep the normal configuration path.
    """
    if set_page_config:
        st.set_page_config(page_title="Credit Risk Advisor", page_icon="🏦")

    st.title("Credit Risk Advisor")
    st.caption(
        "Grounded answers from the Basel III standardised approach for credit risk. "
        "Educational demo — not legal, regulatory, or compliance advice."
    )

    with st.sidebar:
        st.header("Settings")
        search_mode = st.selectbox(
            "Retrieval mode",
            options=list(SEARCH_MODES),
            index=list(SEARCH_MODES).index(DEFAULT_SEARCH_MODE),
            help="rerank won the retrieval evaluation (hit rate 0.911 / MRR 0.768) "
            "and is the default; the others are kept for comparison.",
        )
        use_rewrite = st.checkbox(
            "Rewrite query into Basel terminology",
            value=False,
            help="Expands shorthand like 'CCF for trade LCs' into framework wording "
            "before retrieval. Off by default: on already-well-formed questions the "
            "evaluation showed it reduces hit rate (0.911 → 0.756) and adds ~1.4s "
            "— see the README's Evaluation section.",
        )
        st.markdown(
            "Example questions:\n"
            "- What is the risk weight for an unrated corporate exposure?\n"
            "- What collateral is eligible for credit risk mitigation?\n"
            "- How are exposures to SMEs treated?"
        )

    question = st.text_input("Ask a question about the Basel III credit-risk framework:")

    if st.button("Ask", type="primary") and question.strip():
        request_started = time.perf_counter()
        with st.spinner("Retrieving regulatory passages and generating an answer..."):
            rewrite: RewriteMetrics | None = None
            if use_rewrite:
                start = time.perf_counter()
                rewritten, usage = rewrite_query_with_usage(question)
                rewrite = RewriteMetrics(
                    query=rewritten,
                    tokens=usage.total_tokens,
                    cost=calculate_cost(usage),
                    time=time.perf_counter() - start,
                )
            result: AnswerResult = answer(
                rewrite["query"] if rewrite else question,
                search_mode=search_mode,
            )
            conversation_id = save_conversation(question, result, rewrite=rewrite)
            judge_tokens, judge_cost, judge_time = 0, 0.0, 0.0
            relevance: str | None = None
            explanation: str | None = None
            judge_started = time.perf_counter()
            try:
                relevance, explanation, judge_tokens, judge_cost = evaluate_relevance(
                    question, result["answer"]
                )
                judge_time = time.perf_counter() - judge_started
                save_feedback(
                    conversation_id,
                    "judge",
                    relevance=relevance,
                    explanation=explanation,
                )
            except Exception:  # noqa: BLE001 - judging must never break the app
                relevance, explanation = None, None
                judge_time = time.perf_counter() - judge_started

            total_response_time = time.perf_counter() - request_started
            update_conversation_judge(
                conversation_id,
                judge_tokens,
                judge_cost,
                judge_time=judge_time,
                response_time=total_response_time,
            )
            result["response_time"] = total_response_time

        st.session_state["last"] = LastInteraction(
            conversation_id=conversation_id,
            question=question,
            rewrite=rewrite,
            result=result,
            relevance=relevance,
            explanation=explanation,
            voted=False,
        )

    last = cast(Optional[LastInteraction], st.session_state.get("last"))
    if last:
        result = last["result"]
        st.markdown(f"**Q:** {last['question']}")
        if last["rewrite"] is not None:
            rw = last["rewrite"]
            st.caption(
                f"Query rewritten to: {rw['query']} "
                f"(+{rw['time']:.1f}s, {rw['tokens']} tokens, ${rw['cost']:.4f})"
            )
        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander(f"Sources ({len(result['sources'])} passages retrieved)"):
                for c in result["sources"]:
                    st.markdown(
                        f"**{c['source_title']}, p.{c['page']}** (`{c['chunk_id']}`)"
                    )
                    st.caption(
                        c["text"][:400] + ("..." if len(c["text"]) > 400 else "")
                    )

        meta = (
            f"mode: {result['search_mode']} · {result['response_time']:.1f}s end to end · "
            f"{result['total_tokens']} answer tokens · ${result['cost']:.4f} answer cost"
        )
        if last["relevance"]:
            meta += f" · judge: {last['relevance']}"
        st.caption(meta)

        if not last["voted"]:
            col1, col2, _ = st.columns([1, 1, 6])
            if col1.button("👍 Helpful"):
                save_feedback(last["conversation_id"], "user", score=1)
                last["voted"] = True
                st.session_state["last"] = last
                st.rerun()
            if col2.button("👎 Not helpful"):
                save_feedback(last["conversation_id"], "user", score=-1)
                last["voted"] = True
                st.session_state["last"] = last
                st.rerun()
        else:
            st.caption("Thanks for the feedback!")


if __name__ == "__main__":
    render_app()
