# Hybrid RAG KB - Technical Documentation

This document is split into two parts:
- A terminal-first runbook so you can operate the project immediately.
- A deep dive into internals so you understand how the system works.

## What this project is
Hybrid RAG KB is a FastAPI service that ingests documents, builds a hybrid
retrieval index (BM25 over SQLite FTS5 + vector similarity via FAISS), and
answers questions with citations enforced per paragraph using an Ollama chat
model. It also stores lightweight chat memory in SQLite to provide short
contextual continuity per session.

It includes a **Rich-based CLI** for interactive management, querying, and system health monitoring, making it a complete local knowledge base solution.

Key characteristics:
- Hybrid retrieval: BM25 (sparse) + FAISS (dense) fused with RRF.
- Persisted storage: SQLite for documents/chunks and FAISS for vectors.
- Citation enforcement: answers must cite allowed chunk IDs per paragraph.
- Local LLM runtime: ChatOllama + OllamaEmbeddings on a running Ollama server.
- **Interactive CLI**: Managing ingestion, models, and queries without raw `curl` commands.

## 0) Assumptions (terminal runbook)

- You are in the project root (example below).
- You manage dependencies with `uv` (project mode with `pyproject.toml`).
- You run a local Ollama server, accessed via HTTP.
- You have CUDA 12 + GPU if using `faiss-gpu-cu12`. If not, switch to CPU.

```bash
cd /projects/hybrid-rag-kb
```

## 1) Install dependencies (uv)

### A) Create or sync the venv from `pyproject.toml`
```bash
uv sync
```
What it does:
- Resolves dependencies from `pyproject.toml` and `uv.lock`.
- Creates `.venv` if it does not exist.
- Installs all packages into that venv.

### B) Run commands without activating the venv
```bash
uv run python --version
uv run uvicorn --help
uv run hrag --help
```
What it does:
- `uv run` runs the command inside the project venv automatically.

### C) Optional: activate the venv manually
```bash
source .venv/bin/activate
python --version
```
If the venv is active, you can run commands without prefixing `uv run`.

## 2) Start Ollama and prepare models

### A) Start the Ollama server
```bash
ollama serve
```
This starts the local Ollama HTTP server the API will call.

### B) Download models
These must match your `.env` values. Defaults in this repo:
```bash
ollama pull gemma3:1b
ollama pull mxbai-embed-large
```

### C) Verify installed models
```bash
ollama list
```

### D) Optional: change the Ollama host binding
```bash
export OLLAMA_HOST=0.0.0.0:11434
```
Use this if you want Ollama accessible outside localhost.

## 3) Start the FastAPI server

### Option A) CLI (Recommended)
The CLI includes a command to start the server in the background or foreground.
```bash
uv run hrag start
```

### Option B) Uvicorn (Manual/Dev)
```bash
uv run uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
```
What each part means:
- `uv run` uses the project venv.
- `uvicorn src.main:app` loads `app` from `src/main.py`.
- `--reload` auto-restarts on code changes (dev only).
- `--host 127.0.0.1` binds locally.
- `--port 8000` selects the port.

### Confirm the server is up
```bash
curl -sS -i http://127.0.0.1:8000/docs | head
```
If you see HTML headers, the server is running.

You can also hit health:
```bash
curl -sS http://127.0.0.1:8000/health
```

## 4) Interactive CLI Mode (Preferred)

Instead of using raw `curl` commands, you can launch the interactive CLI:
```bash
uv run hrag
```
This drops you into a REPL where you can:
- `/ingest path/to/docs/`
- `/query "Your question?"
- `/chat` (interactive session)
- `/doctor` (system health check)
- `/model set` (switch models at runtime)

The commands below (Sections 5-7) show the underlying API calls for understanding, but the CLI handles these more gracefully.

## 5) Ingest documents into the KB

Upload one or more files:
```bash
curl -sS \
  -F "files=@/projects/hybrid-rag-kb/ai-consciousness.pdf" \
  -F "files=@/projects/hybrid-rag-kb/ai-cons-11.pdf" \
  http://127.0.0.1:8000/ingest \
| python -m json.tool
```
Notes:
- `-F` sends multipart form data.
- Repeat `-F "files=@..."` for multiple files.
- `python -m json.tool` pretty prints JSON responses.

Check corpus state:
```bash
curl -sS http://127.0.0.1:8000/stats | python -m json.tool
```
If `chunks` equals `faiss.ntotal`, you have one vector per chunk.

## 6) Ask questions with citations

Normal query:
```bash
curl -sS -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","query":"Summarize the main arguments about consciousness.","bm25_k":20,"vec_k":20,"top_k":8,"memory_k":6}' \
| python -m json.tool | head -n 120
```
Notes:
- `session_id` groups chat history in SQLite.
- `memory_k` controls how many recent messages are included.
- The response includes `answer`, `sources`, and `debug`.

## 7) Debug retrieval and citations

### A) Retrieval breakdown (BM25 vs vector vs fused)
```bash
curl -sS -X POST http://127.0.0.1:8000/debug/retrieval \
  -H "Content-Type: application/json" \
  -d '{"query":"Tononi Integrated Information Theory phi","bm25_k":10,"vec_k":10,"top_k":5}' \
| python -m json.tool | head -n 200
```
How to read it:
- `bm25`: FTS5 results ordered by BM25 (lower is better).
- `vector`: FAISS similarity hits.
- `fused`: RRF merge of both ranked lists.

### B) Citation diagnostics
```bash
curl -sS -X POST http://127.0.0.1:8000/debug/citations \
  -H "Content-Type: application/json" \
  -d '{"query":"Explain IIT and cite sources.","bm25_mode":"heuristic"}' \
| python -m json.tool | head -n 120
```
This returns citation validation details and the model answer.

### C) Terminal citation validator (extra safety check)
```bash
curl -sS -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","query":"Summarize the key points and cite sources."}' \
| python -c 'import sys, json, re
data = json.load(sys.stdin)
answer = data.get("answer","")
sources = data.get("sources",[])
cids = sorted({int(x) for x in re.findall(r"cid:(\d+)", answer)})
allowed = sorted({int(s["chunk_id"]) for s in sources if "chunk_id" in s})
print("citations_found:", cids)
print("allowed_chunk_ids_from_sources:", allowed[:30], ("..." if len(allowed)>30 else ""))
if not cids: raise SystemExit("FAIL: No citations found in answer")
invalid = [cid for cid in cids if cid not in set(allowed)]
if invalid: raise SystemExit(f"FAIL: Invalid citations: {invalid}")
print("OK")'
```
This checks that every citation in the answer appears in `sources`.

### D) Direct SQLite BM25 test (optional)
```bash
sqlite3 data/db/app.db "
SELECT c.id, d.filename, bm25(chunks_fts) AS score
FROM chunks_fts
JOIN chunks c ON chunks_fts.rowid = c.id
JOIN documents d ON c.document_id = d.id
WHERE chunks_fts MATCH 'Tononi'
ORDER BY bm25(chunks_fts)
LIMIT 5;"
```
This proves the BM25 channel works independently of FAISS.

## 8) Inspect data on disk

SQLite tables:
```bash
sqlite3 data/db/app.db ".tables"
sqlite3 data/db/app.db "SELECT COUNT(*) FROM documents;"
sqlite3 data/db/app.db "SELECT COUNT(*) FROM chunks;"
sqlite3 data/db/app.db "SELECT COUNT(*) FROM chunks_fts;"
```

FAISS index file:
```bash
ls -lh data/index/faiss/index.faiss
```

## 9) Reset and rebuild (clean slate)

Stop the FastAPI server (Ctrl+C), then:
```bash
rm -f data/db/app.db
rm -f data/index/faiss/index.faiss
rm -rf data/raw
```
Or via CLI:
```bash
/reset
/restart
```

Restart the server and re-ingest:
```bash
uv run uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
```

## 10) Common terminal pitfalls

### `python -m json.tool` fails with "Expecting value"
This usually means the response body was not JSON (server down or error).
Inspect the raw response:
```bash
curl -sS -i http://127.0.0.1:8000/stats
```

### Ollama not reachable
Check if Ollama is running:
```bash
ollama list
```
If it is not, start it:
```bash
ollama serve
```

## Configuration (.env)

`src/core/config.py` reads `.env` and environment variables.

Example `.env`:
```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=gemma3:1b
OLLAMA_EMBED_MODEL=mxbai-embed-large
OLLAMA_NUM_PREDICT=512

SQLITE_PATH=./data/db/app.db
SCHEMA_PATH=./src/db/schema.sql
RAW_DIR=./data/raw
FAISS_DIR=./data/index/faiss

CHUNK_SIZE=1000
CHUNK_OVERLAP=150

USE_FAISS_GPU=true
FAISS_GPU_DEVICE=0
```

## Repo layout and file roles

Top level:
- `src/`: FastAPI app and RAG pipeline.
- `cli/`: Rich-based interactive terminal interface.
- `data/`: Persisted runtime artifacts (SQLite DB, FAISS index, raw files).
- `assets/`: Static assets (images, etc.).
- `main.py`: Simple hello-world stub, not used by the API.
- `pyproject.toml`: Project metadata and dependencies.
- `uv.lock`: Locked dependencies for uv.
- `README.md`: Project overview and usage guide.
- `ai-cons-11.pdf`, `ai-consciousness.pdf`: Sample PDFs.

Core application:
- `src/main.py`: FastAPI app, lifespan hook initializes SQLite schema.
- `src/core/config.py`: Settings loaded from `.env` or environment variables.
- `src/api/routes*.py`: API endpoints. `routes_models.py` handles runtime model switching.
- `src/db/schema.sql`: SQLite schema, including FTS5 and triggers.
- `src/db/sqlite.py`: SQLite connection and schema initialization.

CLI components (`cli/`):
- `cli/main.py`: Entry point, REPL loop, and Rich `Console` setup.
- `cli/core/api_client.py`: `APIClient` wrapper for `httpx` calls to the backend.
- `cli/commands/`: Individual command implementations (`ingest`, `query`, `doctor`, etc.).
- `cli/ui/`: Rich UI components (spinners, panels, themes).

RAG modules:
- `src/rag/ingest.py`: Upload ingestion, dedup, chunking, embedding, FAISS add.
- `src/rag/qa.py`: Hybrid retrieval, fusion, LLM call, citation validation.
- `src/rag/bm25_fts.py`: FTS5 BM25 search and query heuristics.
- `src/rag/vectorstore.py`: FAISS index manager with GPU cloning.
- `src/rag/chunking.py`: Text chunking via RecursiveCharacterTextSplitter.
- `src/rag/loaders.py`: PDF and text extraction.
- `src/rag/hybrid_fusion.py`: Reciprocal Rank Fusion (RRF).
- `src/rag/memory.py`: Persisted chat history in SQLite.
- `src/rag/citations.py`: Citation extraction and validation.

Other folders:
- `scripts/`: Placeholder package (empty).
- `tests/`: Placeholder package (empty).

## CLI Implementation & Architecture

The CLI is a distinct application from the API server but is designed to work in tandem.

**Architecture:**
1. **Interactive REPL**: Built with `prompt_toolkit` for history and `rich` for rendering.
2. **API Client**: `cli/core/api_client.py` abstracts all HTTP calls. It handles connection errors gracefully (e.g., if the server isn't running).
3. **Command Pattern**: Each command (`/ingest`, `/doctor`) is a class in `cli/commands/` inheriting from `BaseCommand`.
4. **State Management**: The CLI relies on the server for RAG state but maintains its own configuration (host/port) and command history.

**The Doctor Command (`/doctor`):**
Implemented in `cli/commands/doctor.py`, this tool performs a "physical" on the system:
- **Dependencies**: Imports critical packages (`faiss`, `langchain`) to verify installation.
- **Ollama Connectivity**: Pings `http://localhost:11434`.
- **Database Integrity**: Connects directly to `data/db/app.db` (read-only) to count rows, verifying the schema exists.
- **FAISS Status**: loads `index.faiss` to check vector counts and dimensions.
- **Animation**: Uses TTY detection to show animations only when running in an interactive terminal.

## How ingestion works (deep flow)

File: `src/rag/ingest.py`

Steps per uploaded file:
1) Read the entire file into memory (`UploadFile.read()`).
2) Compute SHA256 digest and skip duplicates if the hash exists in `documents`.
3) Save raw file to `data/raw/<sha256>_<original_name>`.
4) Insert document row into `documents`.
5) Extract text:
   - PDF: `pypdf.PdfReader` with `page.extract_text()`.
   - Other: UTF-8 decode with `errors="ignore"`.
6) Chunk text using `RecursiveCharacterTextSplitter`.
7) Insert chunks into `chunks` (FTS triggers populate `chunks_fts`).
8) Embed all chunk texts using Ollama embeddings.
9) Add vectors to FAISS, persisting `index.faiss` to disk.

Nuances:
- Deduplication is content-based (SHA256), not filename-based.
- PDF extraction may return empty strings for scanned pages.
- Ingestion reads full files into memory (not streaming).
- SQLite commit happens before vector insertion. If embedding fails, the DB and
  FAISS index can be temporarily out of sync.

## How retrieval and answering work

File: `src/rag/qa.py`

Pipeline:
1) Persist user message into `chat_messages` (if `store_memory=True`).
2) BM25 search via SQLite FTS5 (`bm25_fts.py`), using a heuristic query builder.
3) Embed the query and run FAISS similarity search.
4) Join FAISS results back to chunk records by ID.
5) Fuse BM25 and vector rankings using RRF (`hybrid_fusion.py`).
6) Build a context string with chunk text and citation tokens.
7) Invoke ChatOllama with a system prompt requiring citations per paragraph.
8) Validate citations and optionally rewrite or inject missing citations.
9) Persist assistant response to `chat_messages` (if enabled).

Nuances:
- FAISS uses `IndexFlatIP` with L2-normalized vectors to approximate cosine.
- FAISS IDs map directly to SQLite `chunks.id` via `IndexIDMap2`.
- `bm25_mode` controls tokenization and stopword filtering.
- The citation validator supports both `[cid:123]` and `[Source: ... cid:123]`.
- **Runtime Model Switching**: When models are changed via `/model`, the server updates the global `ChatOllama` and `OllamaEmbeddings` instances in memory. This affects all subsequent requests but does **not** re-process existing data.

## Database schema highlights

File: `src/db/schema.sql`

Tables:
- `documents`: metadata for each uploaded document.
- `chunks`: text chunks with metadata and FK to documents.
- `chunks_fts`: FTS5 virtual table over `chunks.text`.
- `chat_messages`: session-scoped chat history.

FTS triggers keep `chunks_fts` in sync on INSERT/UPDATE/DELETE.

## Operational nuances and gotchas

- **GPU dependency**: `faiss-gpu-cu12` requires CUDA 12. If you are CPU-only,
  switch to `faiss-cpu` and disable GPU in `.env`.
- **Embedding Dimension Mismatch**: If you switch the embedding model (e.g., from `mxbai-embed-large` [1024 dim] to `nomic-embed-text` [768 dim]) via `/model` or `.env`:
    - The FAISS index will still expect 1024 dimensions.
    - Queries will generate 768-dim vectors.
    - **Result**: Dimension mismatch error.
    - **Fix**: You must `/reset` and re-`/ingest` all documents when changing embedding dimensions.
- **Index/DB drift**: Ingestion commits DB before FAISS add. If embedding fails,
  you can end up with chunks that are not indexed. Rebuild index if needed.
- **Large files**: `UploadFile.read()` loads full files into memory.
- **PDF extraction**: Scanned PDFs without text yield empty chunks.
- **Concurrent ingest**: FAISS index writes on every ingest; concurrent writes are
  not coordinated and can corrupt the index.