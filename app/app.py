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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from monitoring.db import save_conversation, save_feedback
from monitoring.judge import evaluate_relevance
from rag.llm import DEFAULT_SEARCH_MODE, answer
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
    st.markdown(
        "Example questions:\n"
        "- What is the risk weight for an unrated corporate exposure?\n"
        "- What collateral is eligible for credit risk mitigation?\n"
        "- How are exposures to SMEs treated?"
    )

question = st.text_input("Ask a question about the Basel III credit-risk framework:")

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Retrieving regulatory passages and generating an answer..."):
        result = answer(question, search_mode=search_mode)
        conversation_id = save_conversation(question, result)
        try:
            relevance, explanation = evaluate_relevance(question, result["answer"])
            save_feedback(conversation_id, "judge",
                          relevance=relevance, explanation=explanation)
        except Exception:  # noqa: BLE001 - judging must never break the app
            relevance, explanation = None, None
    st.session_state.last = {
        "conversation_id": conversation_id,
        "question": question,
        "result": result,
        "relevance": relevance,
        "explanation": explanation,
        "voted": False,
    }

last = st.session_state.get("last")
if last:
    result = last["result"]
    st.markdown(f"**Q:** {last['question']}")
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
