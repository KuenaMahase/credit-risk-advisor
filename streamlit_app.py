"""
Streamlit Community Cloud entrypoint.

Community Cloud runs this file by default. It differs from `streamlit run
app/app.py` (the local/Docker path) only in that it first runs the cold-start
bootstrap: the Docker image bakes the knowledge base in and reads the API key
from an env file, but a fresh Community Cloud container has neither, so the key
is mirrored out of the platform's Secrets and the knowledge base is built on
first boot. Then the unmodified app is executed.

Local and Docker deployments keep using app/app.py directly; this wrapper is
only for hosts that don't run the Docker build.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Credit Risk Advisor", page_icon="🏦")

from app.bootstrap import bootstrap, knowledge_base_ready  # noqa: E402


@st.cache_resource(show_spinner=False)
def prepare_runtime() -> None:
    """Run the one-time, process-wide cloud bootstrap."""
    bootstrap()


if knowledge_base_ready():
    prepare_runtime()
else:
    with st.spinner("First-time setup: downloading the source document and "
                    "building the knowledge base (one-time, ~1 minute)…"):
        prepare_runtime()

from app.app import render_app  # noqa: E402

render_app(set_page_config=False)
