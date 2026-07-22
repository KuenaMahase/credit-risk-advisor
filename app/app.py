"""
Streamlit UI for the Credit Risk Advisor.

Run from the repo root:
    streamlit run app/app.py

Flow per question: retrieve + generate a cited answer (rag.llm.answer) ->
log the conversation to the monitoring store -> auto-judge relevance
(online LLM judge, saved as feedback source='judge') -> offer the user
thumbs up/down (saved as feedback source='user').
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from monitoring.db import save_conversation, save_feedback, update_conversation_judge
from monitoring.judge import evaluate_relevance
from rag.llm import (
    DEFAULT_SEARCH_MODE,
    PRICE_PER_M_INPUT,
    PRICE_PER_M_OUTPUT,
    answer,
    rewrite_query_with_usage,
)
from rag.search import SEARCH_MODES

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
    with st.spinner("Retrieving regulatory passages and generating an answer..."):
        rewrite = None
        if use_rewrite:
            start = time.perf_counter()
            rewritten, usage = rewrite_query_with_usage(question)
            rewrite = {
                "query": rewritten,
                "tokens": usage.total_tokens,
                "cost": (usage.input_tokens * PRICE_PER_M_INPUT
                         + usage.output_tokens * PRICE_PER_M_OUTPUT) / 1_000_000,
                "time": time.perf_counter() - start,
            }
        result = answer(rewrite["query"] if rewrite else question, search_mode=search_mode)
        conversation_id = save_conversation(question, result, rewrite=rewrite)
        try:
            relevance, explanation, judge_tokens, judge_cost = evaluate_relevance(
                question, result["answer"])
            save_feedback(conversation_id, "judge",
                          relevance=relevance, explanation=explanation)
            update_conversation_judge(conversation_id, judge_tokens, judge_cost)
        except Exception:  # noqa: BLE001 - judging must never break the app
            relevance, explanation = None, None
    st.session_state.last = {
        "conversation_id": conversation_id,
        "question": question,
        "rewrite": rewrite,
        "result": result,
        "relevance": relevance,
        "explanation": explanation,
        "voted": False,
    }

last = st.session_state.get("last")
if last:
    result = last["result"]
    st.markdown(f"**Q:** {last['question']}")
    if last.get("rewrite"):
        rw = last["rewrite"]
        st.caption(f"Query rewritten to: {rw['query']} "
                   f"(+{rw['time']:.1f}s, {rw['tokens']} tokens, ${rw['cost']:.4f})")
    st.markdown(result["answer"])

    if result["sources"]:
        with st.expander(f"Sources ({len(result['sources'])} passages retrieved)"):
            for c in result["sources"]:
                st.markdown(f"**{c['source_title']}, p.{c['page']}** (`{c['chunk_id']}`)")
                st.caption(c["text"][:400] + ("..." if len(c["text"]) > 400 else ""))

    meta = (
        f"mode: {result['search_mode']} · {result['response_time']:.1f}s · "
        f"{result['total_tokens']} tokens · ${result['cost']:.4f}"
    )
    if last["relevance"]:
        meta += f" · judge: {last['relevance']}"
    st.caption(meta)

    if not last["voted"]:
        col1, col2, _ = st.columns([1, 1, 6])
        if col1.button("👍 Helpful"):
            save_feedback(last["conversation_id"], "user", score=1)
            st.session_state.last["voted"] = True
            st.rerun()
        if col2.button("👎 Not helpful"):
            save_feedback(last["conversation_id"], "user", score=-1)
            st.session_state.last["voted"] = True
            st.rerun()
    else:
        st.caption("Thanks for the feedback!")
