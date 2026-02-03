# Hybrid RAG KB Technical Documentation

**Document Version:** 2.0  
**Last Updated:** February 2026  
**Project:** Hybrid Retrieval Augmented Generation Knowledge Base

---

## Document Overview

This document is organized into two major parts:

1. **Terminal First Runbook**: Operational guide for immediate project deployment and management
2. **Deep Technical Reference**: Comprehensive analysis of system internals, algorithms, and architecture

---

## Table of Contents

1. [What This Project Is](#what-this-project-is)
2. [System Architecture Overview](#system-architecture-overview)
3. [Terminal Runbook](#terminal-runbook)
   - [Assumptions](#0-assumptions-terminal-runbook)
   - [Install Dependencies](#1-install-dependencies-uv)
   - [Start Ollama and Prepare Models](#2-start-ollama-and-prepare-models)
   - [Start the FastAPI Server](#3-start-the-fastapi-server)
   - [Interactive CLI Mode](#4-interactive-cli-mode-preferred)
   - [Ingest Documents](#5-ingest-documents-into-the-kb)
   - [Ask Questions with Citations](#6-ask-questions-with-citations)
   - [Debug Retrieval and Citations](#7-debug-retrieval-and-citations)
   - [Inspect Data on Disk](#8-inspect-data-on-disk)
   - [Reset and Rebuild](#9-reset-and-rebuild-clean-slate)
   - [Common Terminal Pitfalls](#10-common-terminal-pitfalls)
4. [Configuration Reference](#configuration-env)
5. [Repository Layout and File Roles](#repo-layout-and-file-roles)
6. [CLI Implementation and Architecture](#cli-implementation--architecture)
7. [Core Algorithms Deep Dive](#core-algorithms-deep-dive)
   - [Reciprocal Rank Fusion](#reciprocal-rank-fusion-rrf)
   - [BM25 Scoring Function](#bm25-scoring-function)
   - [Vector Similarity via Inner Product](#vector-similarity-via-inner-product)
8. [Data Structures and Storage Architecture](#data-structures-and-storage-architecture)
   - [SQLite Database Schema](#sqlite-database-schema-analysis)
   - [FAISS Index Architecture](#faiss-index-architecture)
   - [File System Layout](#file-system-layout)
9. [Ingestion Pipeline Deep Flow](#how-ingestion-works-deep-flow)
10. [Retrieval and Answering Pipeline](#how-retrieval-and-answering-work)
11. [Citation Validation System](#citation-validation-system)
12. [Memory and Chat History Management](#memory-and-chat-history-management)
13. [Configuration System Deep Dive](#configuration-system-deep-dive)
14. [Performance Considerations](#performance-considerations)
15. [Security Considerations](#security-considerations)
16. [Operational Nuances and Gotchas](#operational-nuances-and-gotchas)

---

## What This Project Is

Hybrid RAG KB is a FastAPI service that ingests documents, builds a hybrid retrieval index (BM25 over SQLite FTS5 combined with vector similarity via FAISS), and answers questions with citations enforced per paragraph using an Ollama chat model. It also stores lightweight chat memory in SQLite to provide short contextual continuity per session.

It includes a **Rich based CLI** for interactive management, querying, and system health monitoring, making it a complete local knowledge base solution.

### Key Characteristics

| Characteristic | Implementation Details |
|----------------|----------------------|
| Hybrid Retrieval | BM25 (sparse) combined with FAISS (dense) fused with Reciprocal Rank Fusion |
| Persisted Storage | SQLite for documents and chunks, FAISS for vector embeddings |
| Citation Enforcement | Answers must cite allowed chunk IDs per paragraph |
| Local LLM Runtime | ChatOllama combined with OllamaEmbeddings on a running Ollama server |
| Interactive CLI | Managing ingestion, models, and queries without raw curl commands |

---

## System Architecture Overview

### High Level Component Architecture

The system is organized into four distinct layers that work together to provide the complete RAG pipeline:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │   Rich CLI      │    │   HTTP Client   │    │   Swagger UI    │ │
│  │   (Interactive) │    │   (curl/httpx)  │    │   (/docs)       │ │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘ │
└───────────┼──────────────────────┼──────────────────────┼───────────┘
            │                      │                      │
            └──────────────────────┼──────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        API LAYER (FastAPI)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │ /ingest     │  │ /query      │  │ /stats      │  │ /debug/*   │ │
│  │ routes      │  │ routes      │  │ routes      │  │ routes     │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬──────┘ │
└─────────┼────────────────┼────────────────┼───────────────┼─────────┘
          │                │                │               │
          └────────────────┼────────────────┴───────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     RAG PIPELINE LAYER                               │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────────────┐│
│  │ ingest.py │  │ qa.py     │  │ citations │  │ hybrid_fusion.py  ││
│  │           │  │           │  │ .py       │  │                   ││
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────────┬─────────┘│
│        │              │              │                  │          │
│  ┌─────┴─────┐  ┌─────┴─────┐  ┌─────┴─────────────────┴────────┐ │
│  │chunking.py│  │bm25_fts.py│  │        vectorstore.py          │ │
│  │loaders.py │  │memory.py  │  │     (FaissIndexManager)        │ │
│  └───────────┘  └───────────┘  └────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      STORAGE LAYER                                   │
│  ┌─────────────────────────┐    ┌─────────────────────────────────┐│
│  │      SQLite Database    │    │        FAISS Index              ││
│  │  ┌─────────────────┐    │    │  ┌───────────────────────────┐  ││
│  │  │ documents       │    │    │  │ IndexIDMap2(IndexFlatIP)  │  ││
│  │  │ chunks          │    │    │  │ L2 normalized vectors     │  ││
│  │  │ chunks_fts(FTS5)│    │    │  │ Direct ID mapping         │  ││
│  │  │ chat_messages   │    │    │  └───────────────────────────┘  ││
│  │  └─────────────────┘    │    │                                 ││
│  └─────────────────────────┘    └─────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   EXTERNAL SERVICES                                  │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    Ollama Server                                 ││
│  │  ┌─────────────────────┐    ┌─────────────────────────────────┐ ││
│  │  │ Chat Model          │    │ Embedding Model                 │ ││
│  │  │ (gemma3:1b)         │    │ (mxbai_embed_large, 1024 dim)   │ ││
│  │  └─────────────────────┘    └─────────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow: Ingestion Pipeline

```
Document Upload
      │
      ▼
┌─────────────────┐
│ Read file bytes │
│ into memory     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Compute SHA256  │────▶│ Check duplicate │
│ hash            │     │ in documents    │
└─────────────────┘     └────────┬────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
              [EXISTS]                   [NEW FILE]
                    │                         │
                    ▼                         ▼
              Return skip            ┌─────────────────┐
                                     │ Save to raw_dir │
                                     │ with SHA256     │
                                     │ prefix          │
                                     └────────┬────────┘
                                              │
                                              ▼
                                     ┌─────────────────┐
                                     │ Insert document │
                                     │ row (metadata)  │
                                     └────────┬────────┘
                                              │
                                              ▼
                                     ┌─────────────────┐
                                     │ Extract text    │
                                     │ (PDF or UTF8)   │
                                     └────────┬────────┘
                                              │
                                              ▼
                                     ┌─────────────────┐
                                     │ Chunk text      │
                                     │ (Recursive      │
                                     │ Splitter)       │
                                     └────────┬────────┘
                                              │
                                              ▼
                                     ┌─────────────────┐
                                     │ Insert chunks   │
                                     │ (triggers FTS5) │
                                     └────────┬────────┘
                                              │
                                              ▼
                                     ┌─────────────────┐
                                     │ DB COMMIT       │
                                     └────────┬────────┘
                                              │
                                              ▼
                                     ┌─────────────────┐
                                     │ Embed chunks    │
                                     │ via Ollama      │
                                     └────────┬────────┘
                                              │
                                              ▼
                                     ┌─────────────────┐
                                     │ Add to FAISS    │
                                     │ (normalize L2)  │
                                     └────────┬────────┘
                                              │
                                              ▼
                                     ┌─────────────────┐
                                     │ Persist index   │
                                     │ to disk         │
                                     └─────────────────┘
```

### Data Flow: Query Pipeline

```
User Query
      │
      ▼
┌─────────────────┐
│ Store in chat   │
│ memory (if on)  │
└────────┬────────┘
         │
         ├─────────────────────────────────────┐
         │                                     │
         ▼                                     ▼
┌─────────────────┐                   ┌─────────────────┐
│ BM25 CHANNEL    │                   │ VECTOR CHANNEL  │
│                 │                   │                 │
│ Tokenize query  │                   │ Embed query     │
│ Filter stopwords│                   │ via Ollama      │
│ Build FTS5 query│                   │                 │
│ Execute MATCH   │                   │ Normalize L2    │
│ Rank by bm25()  │                   │ Search FAISS    │
└────────┬────────┘                   └────────┬────────┘
         │                                     │
         │      ┌─────────────────┐            │
         └─────▶│ RRF FUSION      │◀───────────┘
                │                 │
                │ Merge by rank   │
                │ Score = Σ w/(k+r)│
                │ Sort descending │
                │ Take top_k      │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Build context   │
                │ with cid tokens │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Load chat       │
                │ history         │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Invoke LLM      │
                │ (ChatOllama)    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Clean answer    │
                │ (remove LLM     │
                │ artifacts)      │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Validate        │
                │ citations       │
                └────────┬────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
         [VALID]               [INVALID]
              │                     │
              │                     ▼
              │            ┌─────────────────┐
              │            │ Inject missing  │
              │            │ citations       │
              │            │ Fix invalid IDs │
              │            └────────┬────────┘
              │                     │
              └──────────┬──────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Store assistant │
                │ response        │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Return answer,  │
                │ sources, debug  │
                └─────────────────┘
```

---

## Terminal Runbook

### 0) Assumptions (terminal runbook)

Before proceeding with the terminal commands, ensure the following conditions are met:

| Assumption | Details |
|------------|---------|
| Working Directory | You are in the project root directory |
| Package Manager | You manage dependencies with uv (project mode with pyproject.toml) |
| Ollama Server | You run a local Ollama server, accessed via HTTP |
| GPU Requirements | You have CUDA 12 combined with GPU if using faiss_gpu_cu12, otherwise switch to CPU |

```bash
cd /projects/hybrid-rag-kb
```

### 1) Install dependencies (uv)

#### A) Create or sync the venv from pyproject.toml

```bash
uv sync
```

What it does:

| Action | Description |
|--------|-------------|
| Dependency Resolution | Resolves dependencies from pyproject.toml and uv.lock |
| Virtual Environment | Creates .venv if it does not exist |
| Package Installation | Installs all packages into that venv |

#### B) Run commands without activating the venv

```bash
uv run python __version__
uv run uvicorn __help
uv run hrag __help
```

What it does: The `uv run` command runs the specified command inside the project venv automatically.

#### C) Optional: activate the venv manually

```bash
source .venv/bin/activate
python __version__
```

If the venv is active, you can run commands without prefixing uv run.

### 2) Start Ollama and prepare models

#### A) Start the Ollama server

```bash
ollama serve
```

This starts the local Ollama HTTP server the API will call.

#### B) Download models

These must match your .env values. Defaults in this repo:

```bash
ollama pull gemma3:1b
ollama pull mxbai_embed_large
```

#### C) Verify installed models

```bash
ollama list
```

#### D) Optional: change the Ollama host binding

```bash
export OLLAMA_HOST=0.0.0.0:11434
```

Use this if you want Ollama accessible outside localhost.

### 3) Start the FastAPI server

#### Option A) CLI (Recommended)

The CLI includes a command to start the server in the background or foreground.

```bash
uv run hrag start
```

#### Option B) Uvicorn (Manual/Dev)

```bash
uv run uvicorn src.main:app __reload __host 127.0.0.1 __port 8000
```

What each part means:

| Component | Purpose |
|-----------|---------|
| uv run | Uses the project venv |
| uvicorn src.main:app | Loads app from src/main.py |
| __reload | Auto restarts on code changes (dev only) |
| __host 127.0.0.1 | Binds locally |
| __port 8000 | Selects the port |

#### Confirm the server is up

```bash
curl _sS _i http://127.0.0.1:8000/docs | head
```

If you see HTML headers, the server is running.

You can also hit health:

```bash
curl _sS http://127.0.0.1:8000/health
```

### 4) Interactive CLI Mode (Preferred)

Instead of using raw curl commands, you can launch the interactive CLI:

```bash
uv run hrag
```

This drops you into a REPL where you can:

| Command | Action |
|---------|--------|
| /ingest path/to/docs/ | Ingest documents |
| /query "Your question?" | Ask a question |
| /chat | Interactive session |
| /doctor | System health check |
| /model set | Switch models at runtime |

The commands below (Sections 5 through 7) show the underlying API calls for understanding, but the CLI handles these more gracefully.

### 5) Ingest documents into the KB

Upload one or more files:

```bash
curl _sS \
  _F "files=@/projects/hybrid_rag_kb/ai_consciousness.pdf" \
  _F "files=@/projects/hybrid_rag_kb/ai_cons_11.pdf" \
  http://127.0.0.1:8000/ingest \
| python _m json.tool
```

Notes:

| Flag | Purpose |
|------|---------|
| _F | Sends multipart form data |
| files=@... | Repeat for multiple files |
| python _m json.tool | Pretty prints JSON responses |

Check corpus state:

```bash
curl _sS http://127.0.0.1:8000/stats | python _m json.tool
```

If chunks equals faiss.ntotal, you have one vector per chunk.

### 6) Ask questions with citations

Normal query:

```bash
curl _sS _X POST http://127.0.0.1:8000/query \
  _H "Content_Type: application/json" \
  _d '{"session_id":"demo","query":"Summarize the main arguments about consciousness.","bm25_k":20,"vec_k":20,"top_k":8,"memory_k":6}' \
| python _m json.tool | head _n 120
```

Notes:

| Parameter | Purpose |
|-----------|---------|
| session_id | Groups chat history in SQLite |
| memory_k | Controls how many recent messages are included |
| response | Includes answer, sources, and debug |

### 7) Debug retrieval and citations

#### A) Retrieval breakdown (BM25 vs vector vs fused)

```bash
curl _sS _X POST http://127.0.0.1:8000/debug/retrieval \
  _H "Content_Type: application/json" \
  _d '{"query":"Tononi Integrated Information Theory phi","bm25_k":10,"vec_k":10,"top_k":5}' \
| python _m json.tool | head _n 200
```

How to read the output:

| Field | Interpretation |
|-------|----------------|
| bm25 | FTS5 results ordered by BM25 (lower is better) |
| vector | FAISS similarity hits |
| fused | RRF merge of both ranked lists |

#### B) Citation diagnostics

```bash
curl _sS _X POST http://127.0.0.1:8000/debug/citations \
  _H "Content_Type: application/json" \
  _d '{"query":"Explain IIT and cite sources.","bm25_mode":"heuristic"}' \
| python _m json.tool | head _n 120
```

This returns citation validation details and the model answer.

#### C) Terminal citation validator (extra safety check)

```bash
curl _sS _X POST http://127.0.0.1:8000/query \
  _H "Content_Type: application/json" \
  _d '{"session_id":"demo","query":"Summarize the key points and cite sources."}' \
| python _c 'import sys, json, re
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

This checks that every citation in the answer appears in sources.

#### D) Direct SQLite BM25 test (optional)

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

### 8) Inspect data on disk

SQLite tables:

```bash
sqlite3 data/db/app.db ".tables"
sqlite3 data/db/app.db "SELECT COUNT(*) FROM documents;"
sqlite3 data/db/app.db "SELECT COUNT(*) FROM chunks;"
sqlite3 data/db/app.db "SELECT COUNT(*) FROM chunks_fts;"
```

FAISS index file:

```bash
ls _lh data/index/faiss/index.faiss
```

### 9) Reset and rebuild (clean slate)

Stop the FastAPI server (Ctrl+C), then:

```bash
rm _f data/db/app.db
rm _f data/index/faiss/index.faiss
rm _rf data/raw
```

Or via CLI:

```bash
/reset
/restart
```

Restart the server and re ingest:

```bash
uv run uvicorn src.main:app __reload __host 127.0.0.1 __port 8000
```

### 10) Common terminal pitfalls

#### python _m json.tool fails with "Expecting value"

This usually means the response body was not JSON (server down or error).

Inspect the raw response:

```bash
curl _sS _i http://127.0.0.1:8000/stats
```

#### Ollama not reachable

Check if Ollama is running:

```bash
ollama list
```

If it is not, start it:

```bash
ollama serve
```

---

## Configuration (.env)

src/core/config.py reads .env and environment variables.

Example .env:

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=gemma3:1b
OLLAMA_EMBED_MODEL=mxbai_embed_large
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

---

## Repo Layout and File Roles

### Top Level Structure

| Path | Purpose |
|------|---------|
| src/ | FastAPI app and RAG pipeline |
| cli/ | Rich based interactive terminal interface |
| data/ | Persisted runtime artifacts (SQLite DB, FAISS index, raw files) |
| assets/ | Static assets (images, etc.) |
| main.py | Simple hello world stub, not used by the API |
| pyproject.toml | Project metadata and dependencies |
| uv.lock | Locked dependencies for uv |
| README.md | Project overview and usage guide |

### Core Application

| File | Role |
|------|------|
| src/main.py | FastAPI app, lifespan hook initializes SQLite schema |
| src/core/config.py | Settings loaded from .env or environment variables |
| src/api/routes*.py | API endpoints, routes_models.py handles runtime model switching |
| src/db/schema.sql | SQLite schema, including FTS5 and triggers |
| src/db/sqlite.py | SQLite connection and schema initialization |

### CLI Components (cli/)

| File | Role |
|------|------|
| cli/main.py | Entry point, REPL loop, and Rich Console setup |
| cli/core/api_client.py | APIClient wrapper for httpx calls to the backend |
| cli/commands/ | Individual command implementations (ingest, query, doctor, etc.) |
| cli/ui/ | Rich UI components (spinners, panels, themes) |

### RAG Modules

| File | Role |
|------|------|
| src/rag/ingest.py | Upload ingestion, dedup, chunking, embedding, FAISS add |
| src/rag/qa.py | Hybrid retrieval, fusion, LLM call, citation validation |
| src/rag/bm25_fts.py | FTS5 BM25 search and query heuristics |
| src/rag/vectorstore.py | FAISS index manager with GPU cloning |
| src/rag/chunking.py | Text chunking via RecursiveCharacterTextSplitter |
| src/rag/loaders.py | PDF and text extraction |
| src/rag/hybrid_fusion.py | Reciprocal Rank Fusion (RRF) |
| src/rag/memory.py | Persisted chat history in SQLite |
| src/rag/citations.py | Citation extraction and validation |

### Other Folders

| Path | Purpose |
|------|---------|
| scripts/ | Placeholder package (empty) |
| tests/ | Placeholder package (empty) |

---

## CLI Implementation & Architecture

The CLI is a distinct application from the API server but is designed to work in tandem.

### Architecture Overview

| Component | Description |
|-----------|-------------|
| Interactive REPL | Built with prompt_toolkit for history and rich for rendering |
| API Client | cli/core/api_client.py abstracts all HTTP calls, handles connection errors gracefully |
| Command Pattern | Each command (/ingest, /doctor) is a class in cli/commands/ inheriting from BaseCommand |
| State Management | CLI relies on server for RAG state, maintains its own configuration (host/port) and command history |

### The Doctor Command (/doctor)

Implemented in cli/commands/doctor.py, this tool performs a comprehensive system health check:

| Check | Description |
|-------|-------------|
| Dependencies | Imports critical packages (faiss, langchain) to verify installation |
| Ollama Connectivity | Pings http://localhost:11434 |
| Database Integrity | Connects directly to data/db/app.db (read only) to count rows, verifying the schema exists |
| FAISS Status | Loads index.faiss to check vector counts and dimensions |
| Animation | Uses TTY detection to show animations only when running in an interactive terminal |

---

## Core Algorithms Deep Dive

### Reciprocal Rank Fusion (RRF)

Reciprocal Rank Fusion is a score aggregation technique that combines multiple ranked lists into a single unified ranking. The key insight is that RRF fuses by rank position rather than raw scores, which eliminates the need to normalize scores across different retrieval methods that may produce values on entirely different scales.

#### Mathematical Formula

For a document `d` appearing in multiple result lists:

```
fused_score(d) = Σ weight_i / (k + rank_i(d))
```

Where:

| Symbol | Description | Default Value |
|--------|-------------|---------------|
| d | A document being scored | N/A |
| k | RRF smoothing constant (prevents division by small numbers) | 60 |
| rank_i(d) | 1 indexed rank of document d in result list i | N/A |
| weight_i | Weight assigned to result list i | 1.0 for both BM25 and vector |

#### Implementation Details

The implementation in `src/rag/hybrid_fusion.py` follows this process:

1. **Initialize accumulator**: Create a dictionary keyed by chunk_id
2. **Process BM25 results**: For each result at rank r, add `w_bm25 / (k + r)` to the fused score
3. **Process vector results**: For each result at rank r, add `w_vec / (k + r)` to the fused score
4. **Merge metadata**: If a chunk appears in both lists, combine metadata from both sources
5. **Sort and truncate**: Sort by fused_score descending, return top_k results

#### Why k = 60?

The constant k = 60 is a commonly used value in information retrieval literature. It provides a good balance:

| Consideration | Effect |
|---------------|--------|
| Higher k values | Reduce the impact of rank position differences, making the fusion more uniform |
| Lower k values | Amplify the advantage of top ranked results |
| k = 60 | Empirically shown to work well across diverse retrieval scenarios |

### BM25 Scoring Function

BM25 (Best Matching 25) is a probabilistic ranking function used for full text search. SQLite FTS5 implements BM25 natively.

#### Mathematical Formula

```
BM25(D, Q) = Σ IDF(qi) × f(qi, D) × (k1 + 1) / (f(qi, D) + k1 × (1 - b + b × |D| / avgdl))
```

Where:

| Symbol | Description |
|--------|-------------|
| D | Document being scored |
| Q | Query consisting of terms q1, q2, ... qn |
| IDF(qi) | Inverse Document Frequency of term qi |
| f(qi, D) | Frequency of term qi in document D |
| k1 | Term saturation parameter (controls how quickly frequency gains diminish) |
| b | Length normalization parameter (0 = no normalization, 1 = full normalization) |
| \|D\| | Length of document D in tokens |
| avgdl | Average document length across the corpus |

#### IDF Calculation

```
IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
```

Where:

| Symbol | Description |
|--------|-------------|
| N | Total number of documents in the corpus |
| n(qi) | Number of documents containing term qi |

#### SQLite FTS5 Implementation

SQLite FTS5 provides the `bm25()` function which returns a relevance score. Important notes:

| Characteristic | Behavior |
|----------------|----------|
| Score direction | Lower (more negative) values indicate better matches |
| Token handling | FTS5 uses a simple tokenizer by default |
| Column weighting | Supports per column weights if multiple columns are indexed |

#### Query Preprocessing

The `make_bm25_query` function in `src/rag/bm25_fts.py` preprocesses queries for FTS5:

| Mode | Behavior |
|------|----------|
| raw | Tokenizes by word boundaries, keeps all tokens |
| heuristic | Filters stopwords, removes short tokens (< 3 chars), limits to max_terms |

The stopword list includes approximately 50 common English words plus some query instruction words like "summarize", "cite", and "sources" that rarely appear in documents but frequently appear in user queries.

### Vector Similarity via Inner Product

The system uses FAISS IndexFlatIP (Inner Product) for vector similarity search. By L2 normalizing vectors before storage and search, the inner product becomes equivalent to cosine similarity.

#### Mathematical Foundation

For two vectors a and b:

**Cosine Similarity:**
```
cosine(a, b) = (a · b) / (||a|| × ||b||)
```

**When vectors are L2 normalized** (||a|| = ||b|| = 1):
```
cosine(a, b) = a · b = inner_product(a, b)
```

#### Normalization Process

The `FaissIndexManager` class applies L2 normalization at two points:

| Operation | Normalization Applied |
|-----------|----------------------|
| add() | `faiss.normalize_L2(vectors)` before adding to index |
| search() | `faiss.normalize_L2(query_vector)` before searching |

This ensures all comparisons use cosine similarity semantics.

#### Why IndexFlatIP with IDMap2?

| Choice | Rationale |
|--------|-----------|
| IndexFlatIP | Exact inner product search, no approximation, suitable for moderate corpus sizes |
| IndexIDMap2 | Allows arbitrary 64 bit IDs, enabling direct mapping of SQLite chunk IDs to FAISS |

---

## Data Structures and Storage Architecture

### SQLite Database Schema Analysis

The database schema is defined in `src/db/schema.sql` and consists of four main components:

#### documents Table

Stores metadata for each uploaded document:

| Column | Type | Constraints | Purpose |
|--------|------|------------|---------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT | Unique document identifier |
| filename | TEXT | NOT NULL | Original filename |
| sha256 | TEXT | NOT NULL | Content hash for deduplication |
| content_type | TEXT | nullable | MIME type if provided |
| stored_path | TEXT | NOT NULL | Path to raw file on disk |
| created_at | TEXT | NOT NULL DEFAULT datetime('now') | Insertion timestamp |

#### chunks Table

Stores individual text chunks with foreign key relationship to documents:

| Column | Type | Constraints | Purpose |
|--------|------|------------|---------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT | Unique chunk identifier (maps to FAISS ID) |
| document_id | INTEGER | NOT NULL, FK to documents(id) ON DELETE CASCADE | Parent document reference |
| chunk_index | INTEGER | NOT NULL | Position within the document |
| text | TEXT | NOT NULL | Chunk content |
| metadata_json | TEXT | nullable | JSON encoded metadata |
| created_at | TEXT | NOT NULL DEFAULT datetime('now') | Insertion timestamp |

#### chunks_fts Virtual Table

FTS5 full text search index configured in external content mode:

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
USING fts5(
    text,
    content='chunks',
    content_rowid='id'
)
```

| Configuration | Purpose |
|---------------|---------|
| content='chunks' | External content mode, content stored in chunks table |
| content_rowid='id' | Maps FTS rowid to chunks.id |

#### Synchronization Triggers

Three triggers maintain FTS5 synchronization:

| Trigger | Event | Action |
|---------|-------|--------|
| chunks_ai | AFTER INSERT | Inserts new text into chunks_fts |
| chunks_ad | AFTER DELETE | Issues delete command to chunks_fts |
| chunks_au | AFTER UPDATE | Deletes old entry, inserts new entry |

#### chat_messages Table

Stores session scoped conversation history:

| Column | Type | Constraints | Purpose |
|--------|------|------------|---------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT | Message sequence number |
| session_id | TEXT | NOT NULL | Session grouping key |
| role | TEXT | NOT NULL | "user" or "assistant" |
| content | TEXT | NOT NULL | Message text |
| created_at | TEXT | NOT NULL DEFAULT datetime('now') | Timestamp |

#### Database Indexes

| Index | Columns | Purpose |
|-------|---------|---------|
| idx_chunks_document_id | chunks(document_id) | Fast chunk lookup by document |
| idx_documents_sha256 | documents(sha256) | Fast deduplication check |
| idx_chat_messages_session_id | chat_messages(session_id) | Fast history retrieval |

#### PRAGMA Settings

| Setting | Value | Purpose |
|---------|-------|---------|
| journal_mode | WAL | Write Ahead Logging for concurrent reads during writes |
| foreign_keys | ON | Enables foreign key constraint enforcement |

### FAISS Index Architecture

The FAISS index is managed by `FaissIndexManager` in `src/rag/vectorstore.py`.

#### Index Structure

```
IndexIDMap2
    └── IndexFlatIP (base index)
            └── L2 normalized vectors (float32)
```

| Component | Purpose |
|-----------|---------|
| IndexFlatIP | Exact inner product search (brute force, no quantization) |
| IndexIDMap2 | Wraps base index, allows add_with_ids for custom ID assignment |

#### CPU/GPU Architecture

The system maintains the CPU index as the source of truth:

| Index | Storage | Purpose |
|-------|---------|---------|
| cpu_index | Disk (index.faiss) | Source of truth, persisted after every add() |
| gpu_index | GPU memory (optional) | Fast search copy, recreated from cpu_index |

#### GPU Failover Behavior

When GPU copy fails (e.g., out of memory), the system gracefully falls back:

1. Attempt to clone CPU index to GPU via `faiss.index_cpu_to_gpu()`
2. On RuntimeError (CUDA OOM), log warning and set gpu_index = None
3. Subsequent searches use cpu_index instead

#### ID Mapping Strategy

SQLite chunk IDs are used directly as FAISS IDs:

| Operation | Effect |
|-----------|--------|
| add(ids, vectors) | Calls add_with_ids with int64 version of chunk IDs |
| search(query, k) | Returns int IDs that map directly to chunks.id |

This eliminates the need for a separate ID mapping layer.

### File System Layout

```
data/
├── db/
│   └── app.db              # SQLite database file
├── index/
│   └── faiss/
│       └── index.faiss     # Persisted FAISS index (binary format)
└── raw/
    ├── abc123_document1.pdf    # SHA256 prefixed original files
    └── def456_document2.txt
```

| Directory | Contents | Persistence |
|-----------|----------|-------------|
| data/db/ | SQLite database | Survives restarts |
| data/index/faiss/ | FAISS index file | Survives restarts |
| data/raw/ | Original uploaded files | Survives restarts |

---

## How Ingestion Works (Deep Flow)

File: `src/rag/ingest.py`

### Step by Step Process

For each uploaded file:

| Step | Function | Description |
|------|----------|-------------|
| 1 | File read | Read the entire file into memory (UploadFile.read()) |
| 2 | Hash computation | Compute SHA256 digest of file bytes |
| 3 | Deduplication check | Query documents table for existing hash |
| 4 | Raw file storage | Save to data/raw/{sha256}_{original_name} |
| 5 | Document insert | Insert document row into documents table |
| 6 | Text extraction | Extract text based on file type |
| 7 | Chunking | Split text using RecursiveCharacterTextSplitter |
| 8 | Chunk insert | Insert chunks into chunks table (triggers FTS5) |
| 9 | Database commit | Commit transaction |
| 10 | Embedding | Embed all chunk texts using Ollama embeddings |
| 11 | FAISS add | Add vectors to FAISS with chunk IDs |
| 12 | Index persist | Save FAISS index to disk |

### Text Extraction Methods

| File Type | Method | Library |
|-----------|--------|---------|
| PDF | Page by page text extraction | pypdf.PdfReader |
| Other | UTF8 decode with errors="ignore" | Built in Python |

### Chunking Configuration

The RecursiveCharacterTextSplitter from LangChain is configured with:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| chunk_size | 1000 | Maximum characters per chunk |
| chunk_overlap | 150 | Characters shared between adjacent chunks |

The splitter attempts to split on natural boundaries in order of preference:

1. Double newlines (paragraph boundaries)
2. Single newlines
3. Spaces
4. Characters

### Critical Nuances

| Issue | Implication |
|-------|-------------|
| Deduplication is content based | Same file with different name is detected as duplicate |
| PDF extraction limitations | Scanned PDFs without OCR return empty strings |
| Memory loading | Full file loaded into memory (not streaming) |
| Transaction boundary | DB commit happens before vector insertion |
| Failure recovery | If embedding fails after DB commit, chunks exist without vectors |

---

## How Retrieval and Answering Work

File: `src/rag/qa.py`

### Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | memory.py | Persist user message (if store_memory=True) |
| 2 | bm25_fts.py | BM25 search via SQLite FTS5 |
| 3 | vectorstore.py | Embed query and search FAISS |
| 4 | hybrid_fusion.py | Fuse BM25 and vector rankings via RRF |
| 5 | qa.py | Build context with citation tokens |
| 6 | ChatOllama | Invoke LLM with system prompt and history |
| 7 | qa.py | Clean common LLM artifacts from answer |
| 8 | citations.py | Validate citations, repair if needed |
| 9 | memory.py | Persist assistant response (if enabled) |

### BM25 Query Modes

| Mode | Behavior |
|------|----------|
| raw | Keep all tokens (after safe tokenization), no stopword filtering |
| heuristic | Remove stopwords, short tokens (< 3 chars), limit to max_terms |

### Context Formatting

Each chunk is formatted as:

```
[cid:123] from filename.pdf:
{chunk text content}
```

Chunks are separated by `\n\n ---\n\n` for clear delineation.

### System Prompt Structure

The system prompt instructs the LLM to:

| Instruction | Enforcement |
|-------------|-------------|
| Answer using ONLY provided sources | Prevents hallucination |
| Write 2 to 3 paragraphs | Ensures substantive response |
| End each paragraph with citation | Citation per paragraph |
| Use ONLY provided citation IDs | Prevents fabricated citations |
| Start immediately with content | No preamble |
| Do not copy verbatim | Synthesize in own words |

### Answer Post Processing

The `_clean_answer` function removes common LLM artifacts:

| Artifact Type | Examples |
|---------------|----------|
| Preamble | "Okay, here's...", "Based on the context..." |
| Bibliography | "References:", "Sources:" sections at end |
| Author metadata | Email addresses, institutional affiliations |
| Multiple blank lines | Collapsed to single blank line |

### Runtime Model Switching

When models are changed via `/model`, the server updates the global ChatOllama and OllamaEmbeddings instances in memory. This affects all subsequent requests but does **not** re process existing data.

---

## Citation Validation System

File: `src/rag/citations.py`

### Supported Citation Formats

| Format | Pattern | Example |
|--------|---------|---------|
| Simple | `[cid:NUMBER]` | [cid:42] |
| Human readable | `[Source: filename \| cid:NUMBER]` | [Source: paper.pdf \| cid:42] |

### Regex Patterns

```python
_CID_SIMPLE = r"\[cid:(\d+)\]"
_CID_SOURCE = r"\[Source:[^\]]*?\bcid:(\d+)\b[^\]]*\]"
```

### Validation Checks

| Check | Configurable | Default |
|-------|--------------|---------|
| Minimum unique citations | min_unique_citations | 1 |
| All cited IDs in allowed set | always checked | N/A |
| Citation per paragraph | require_citation_per_paragraph | True |

### Validation Report

The validation function returns a detailed report:

| Field | Description |
|-------|-------------|
| paragraph_count | Number of paragraphs detected |
| found_citations | List of unique citation IDs found |
| unique_citations_count | Count of unique citations |
| invalid_ids | List of citations not in allowed set |
| missing_paragraphs | Indices of paragraphs without citations |
| per_paragraph_citations | List of citation IDs per paragraph |
| reason | "ok" or explanation of failure |

### Citation Repair Mechanisms

| Repair | Trigger | Action |
|--------|---------|--------|
| Injection | missing_paragraphs not empty | Append citation from cite_tokens to paragraph end |
| Replacement | invalid_ids not empty | Replace invalid ID with first valid ID |

---

## Memory and Chat History Management

File: `src/rag/memory.py`

### Session Isolation

Messages are grouped by session_id:

| Operation | Scope |
|-----------|-------|
| add_message | Inserts with provided session_id |
| get_recent_messages | Filters by session_id |

### Message Storage

| Field | Storage |
|-------|---------|
| role | "user" or "assistant" |
| content | Full message text |
| ordering | By ascending id (insertion order) |

### Retrieval Strategy

1. Query most recent N messages by id DESC
2. Reverse to chronological order
3. Map roles to LangChain format ("user" → "human")

### Role Mapping

| Database Role | LangChain Role |
|---------------|----------------|
| user | human |
| assistant | assistant |
| other | system |

---

## Configuration System Deep Dive

File: `src/core/config.py`

### Pydantic Settings Architecture

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf_8")
```

| Feature | Behavior |
|---------|----------|
| BaseSettings inheritance | Automatic environment variable binding |
| SettingsConfigDict | Configures .env file loading |
| env_file_encoding | Specifies UTF8 encoding for .env |

### Configuration Parameters Reference

| Parameter | Environment Variable | Default | Description |
|-----------|---------------------|---------|-------------|
| ollama_base_url | OLLAMA_BASE_URL | http://localhost:11434 | Ollama server URL |
| ollama_chat_model | OLLAMA_CHAT_MODEL | gemma3:1b | Model for chat completion |
| ollama_embed_model | OLLAMA_EMBED_MODEL | mxbai_embed_large | Model for embeddings (1024 dim) |
| ollama_num_predict | OLLAMA_NUM_PREDICT | 512 | Max tokens to generate |
| sqlite_path | SQLITE_PATH | ./data/db/app.db | Database file path |
| schema_path | SCHEMA_PATH | ./src/db/schema.sql | Schema file path |
| raw_dir | RAW_DIR | ./data/raw | Raw file storage directory |
| faiss_dir | FAISS_DIR | ./data/index/faiss | FAISS index directory |
| chunk_size | CHUNK_SIZE | 1000 | Max characters per chunk |
| chunk_overlap | CHUNK_OVERLAP | 150 | Overlap between chunks |
| use_faiss_gpu | USE_FAISS_GPU | true | Enable GPU for FAISS |
| faiss_gpu_device | FAISS_GPU_DEVICE | 0 | GPU device index |

---

## Performance Considerations

### Memory Footprint Analysis

| Component | Memory Impact | Mitigation |
|-----------|---------------|------------|
| Document loading | Full file in memory during ingestion | Process smaller files or implement streaming |
| Embedding batching | Entire chunk list embedded at once | Consider batch size limits for large documents |
| FAISS index | Full index in memory during operations | GPU copy adds additional memory pressure |
| SQLite connection | Minimal (connection pooling) | N/A |

### Latency Factors

| Operation | Typical Latency | Optimization |
|-----------|-----------------|--------------|
| BM25 search | < 1ms for small corpora | FTS5 is highly optimized |
| Vector search (CPU) | 10 to 100ms depending on corpus size | Use GPU |
| Vector search (GPU) | < 10ms for moderate corpus | Ensure GPU memory available |
| LLM inference | 5 to 30 seconds depending on model | Use smaller models (gemma3:1b) |
| Embedding | 100ms to 1s per chunk | Batch processing |

### Scalability Limits

| Constraint | Limit | Workaround |
|------------|-------|------------|
| Concurrent ingestion | Not supported | Queue ingestion requests |
| Corpus size (IndexFlatIP) | Memory bound (millions of vectors) | Consider IVF indexes for larger corpora |
| Large files | 100MB+ may cause memory issues | Pre split large files |
| FAISS write | Single writer | Coordinate ingestion |

---

## Security Considerations

### Local Execution Model

| Aspect | Implementation |
|--------|----------------|
| LLM inference | Runs entirely on local Ollama server |
| Embedding | Local Ollama embeddings, no external API |
| Data storage | All data stored locally in data/ directory |
| Network exposure | Default bind to 127.0.0.1 only |

### Data Isolation

| Data Type | Storage Location | Access Control |
|-----------|------------------|----------------|
| Documents | data/raw/ | File system permissions |
| Chunks | data/db/app.db | SQLite file permissions |
| Embeddings | data/index/faiss/ | File system permissions |
| Chat history | data/db/app.db | Session ID scoping |

### Input Validation

| Input | Validation |
|-------|------------|
| Uploaded files | File type determined by extension |
| Query text | Tokenized and sanitized for FTS5 |
| Session IDs | Used as opaque strings |
| Model names | Validated against Ollama available models |

### SQL Injection Prevention

All database queries use parameterized statements:

```python
await db.execute("SELECT ... WHERE sha256 = ?", (digest,))
```

No string concatenation is used for query building.

---

## Operational Nuances and Gotchas

### GPU Dependency

faiss_gpu_cu12 requires CUDA 12. If you are CPU only, switch to faiss_cpu and disable GPU in .env.

### Embedding Dimension Mismatch

If you switch the embedding model (e.g., from mxbai_embed_large [1024 dim] to nomic_embed_text [768 dim]) via /model or .env:

| Component | Behavior |
|-----------|----------|
| FAISS index | Still expects 1024 dimensions |
| New queries | Generate 768 dim vectors |
| Result | Dimension mismatch error |
| Fix | Must /reset and re /ingest all documents |

### Index/DB Drift

Ingestion commits DB before FAISS add. If embedding fails, you can end up with chunks that are not indexed. Use /doctor to diagnose, rebuild index if needed.

### Large Files

UploadFile.read() loads full files into memory. Consider file size limits for production deployments.

### PDF Extraction

Scanned PDFs without embedded text (image only) yield empty chunks. Consider OCR preprocessing for such documents.

### Concurrent Ingest

FAISS index writes on every ingest. Concurrent writes are not coordinated and can corrupt the index. Serialize ingestion requests in production.

---

## Appendix: Quick Reference

### Essential CLI Commands

| Command | Purpose |
|---------|---------|
| /start | Start the FastAPI server |
| /stop | Stop the server |
| /restart | Restart the server |
| /ingest path | Ingest documents |
| /query text | Ask a question |
| /chat | Interactive chat session |
| /doctor | System health check |
| /stats | View corpus statistics |
| /reset | Clear all data |
| /model list | List available models |
| /model set | Interactive model selection |

### Essential API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| /health | GET | Health check |
| /stats | GET | Corpus statistics |
| /ingest | POST | Upload documents |
| /query | POST | Ask questions |
| /models | GET/POST | Model management |
| /debug/retrieval | POST | Retrieval debugging |
| /debug/citations | POST | Citation debugging |

### Environment Variable Quick Reference

| Variable | Required | Purpose |
|----------|----------|---------|
| OLLAMA_BASE_URL | No | Ollama server location |
| OLLAMA_CHAT_MODEL | No | Chat model name |
| OLLAMA_EMBED_MODEL | No | Embedding model name |
| USE_FAISS_GPU | No | Enable/disable GPU |
| CHUNK_SIZE | No | Chunk size in characters |
| CHUNK_OVERLAP | No | Overlap between chunks |