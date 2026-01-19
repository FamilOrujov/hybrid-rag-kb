FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml requirements.txt ./

RUN uv venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
RUN uv pip install --no-cache -r requirements.txt || \
    (sed -i 's/faiss-gpu-cu12==1.8.0.2/faiss-cpu/g' requirements.txt && \
    sed -i '/nvidia-/d' requirements.txt && \
    uv pip install --no-cache -r requirements.txt)


FROM python:3.12-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

COPY src/ ./src/
COPY cli/ ./cli/

RUN mkdir -p data/db data/index/faiss data/raw

ENV OLLAMA_BASE_URL=http://host.docker.internal:11434
ENV SQLITE_PATH=./data/db/app.db
ENV SCHEMA_PATH=./src/db/schema.sql
ENV RAW_DIR=./data/raw
ENV FAISS_DIR=./data/index/faiss
ENV USE_FAISS_GPU=false

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
