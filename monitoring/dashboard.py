"""
Monitoring dashboard for the Credit Risk Advisor.

Run from the repo root:
    streamlit run monitoring/dashboard.py

Reads the SQLite monitoring store (conversations + feedback) and shows
headline metrics plus six charts: query volume, user feedback split,
judge relevance distribution, response time, cumulative cost, and
retrieval-mode usage — with a recent-conversations table as data view.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from monitoring.db import DB_FILE, get_connection, init_db

# Single-hue for neutral quantitative charts; status colors (never reused as
# series colors) for feedback/judge, always paired with visible text labels.
SERIES_BLUE = "#2a78d6"
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_CRITICAL = "#d03b3b"

st.set_page_config(page_title="Credit Risk Advisor — Monitoring", page_icon="📊", layout="wide")
st.title("Credit Risk Advisor — Monitoring")

init_db()
conn = get_connection()
conversations = pd.read_sql_query("SELECT * FROM conversations", conn)
feedback = pd.read_sql_query("SELECT * FROM feedback", conn)
conn.close()

if conversations.empty:
    st.info(f"No conversations logged yet (store: {DB_FILE.name}). "
            "Ask questions in the app first: `streamlit run app/app.py`")
    st.stop()

conversations["timestamp"] = pd.to_datetime(conversations["timestamp"])
user_fb = feedback[feedback["source"] == "user"]
judge_fb = feedback[feedback["source"] == "judge"]

# Headline tiles
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Conversations", len(conversations))
col2.metric("Avg response time", f"{conversations['response_time'].mean():.1f}s")
col3.metric("Total LLM cost", f"${conversations['cost'].sum():.4f}")
col4.metric("Avg tokens / answer", f"{conversations['total_tokens'].mean():.0f}")
thumbs_up_rate = (user_fb["score"] == 1).mean() if len(user_fb) else None
col5.metric("👍 rate", f"{100 * thumbs_up_rate:.0f}%" if thumbs_up_rate is not None else "—")

left, right = st.columns(2)

with left:
    st.subheader("Query volume")
    by_day = conversations.set_index("timestamp").resample("D").size().rename("questions").reset_index()
    fig = px.bar(by_day, x="timestamp", y="questions")
    fig.update_traces(marker_color=SERIES_BLUE)
    fig.update_layout(margin=dict(t=10, b=10), xaxis_title=None, yaxis_title="questions / day")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Judge relevance")
    if judge_fb.empty:
        st.caption("No judge verdicts yet.")
    else:
        order = ["RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"]
        counts = judge_fb["relevance"].value_counts().reindex(order, fill_value=0).reset_index()
        counts.columns = ["relevance", "count"]
        fig = px.bar(counts, x="relevance", y="count", text="count",
                     color="relevance",
                     color_discrete_map={"RELEVANT": STATUS_GOOD,
                                         "PARTLY_RELEVANT": STATUS_WARNING,
                                         "NON_RELEVANT": STATUS_CRITICAL})
        fig.update_layout(margin=dict(t=10, b=10), xaxis_title=None,
                          yaxis_title="answers", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Retrieval-mode usage")
    mode_counts = conversations["search_mode"].value_counts().reset_index()
    mode_counts.columns = ["mode", "count"]
    fig = px.bar(mode_counts, x="mode", y="count", text="count")
    fig.update_traces(marker_color=SERIES_BLUE)
    fig.update_layout(margin=dict(t=10, b=10), xaxis_title=None, yaxis_title="questions")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("User feedback")
    if user_fb.empty:
        st.caption("No user feedback yet.")
    else:
        fb_counts = pd.DataFrame({
            "feedback": ["👍 helpful", "👎 not helpful"],
            "count": [(user_fb["score"] == 1).sum(), (user_fb["score"] == -1).sum()],
        })
        fig = px.bar(fb_counts, x="feedback", y="count", text="count",
                     color="feedback",
                     color_discrete_map={"👍 helpful": STATUS_GOOD,
                                         "👎 not helpful": STATUS_CRITICAL})
        fig.update_layout(margin=dict(t=10, b=10), xaxis_title=None,
                          yaxis_title="votes", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Response time")
    fig = px.line(conversations, x="timestamp", y="response_time", markers=True)
    fig.update_traces(line_color=SERIES_BLUE, marker_color=SERIES_BLUE)
    fig.update_layout(margin=dict(t=10, b=10), xaxis_title=None, yaxis_title="seconds")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cumulative LLM cost")
    cost = conversations.sort_values("timestamp").copy()
    cost["cumulative_cost"] = cost["cost"].cumsum()
    fig = px.line(cost, x="timestamp", y="cumulative_cost", markers=True)
    fig.update_traces(line_color=SERIES_BLUE, marker_color=SERIES_BLUE)
    fig.update_layout(margin=dict(t=10, b=10), xaxis_title=None, yaxis_title="USD")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Recent conversations")
recent = conversations.sort_values("timestamp", ascending=False).head(20)
st.dataframe(
    recent[["timestamp", "question", "search_mode", "total_tokens", "response_time", "cost"]],
    use_container_width=True,
    hide_index=True,
)
