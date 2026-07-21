FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
# Install CPU-only torch first: on linux/arm64 the default PyPI torch pulls
# multi-GB CUDA wheels that are useless in this CPU container. With torch
# already satisfied, the requirements install skips that resolution entirely.
RUN pip install --no-cache-dir "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the knowledge base and warm the retrieval stack at image build time:
# downloads the Basel III PDF, loads the DuckDB KB + chunks.jsonl via the dlt
# pipeline, then runs one rerank query so the sentence-transformer and
# cross-encoder models and the chunk embeddings are all baked into the image.
# `docker compose up` therefore needs no host-side steps and answers fast.
RUN python ingestion/dlt_pipeline.py && \
    python -c "from rag.search import rerank_search; rerank_search('warm-up query')"

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
