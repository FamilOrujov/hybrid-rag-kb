from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from src.api.model_config import get_initial_embed_model
from src.core.config import settings
from src.db.sqlite import connect
from src.rag.bm25_fts import bm25_search, make_bm25_query
from src.rag.embeddings import make_embedder
from src.rag.hybrid_fusion import rrf_fuse
from src.rag.vectorstore import FaissIndexManager

router = APIRouter()

_embedder = make_embedder(settings.ollama_base_url, get_initial_embed_model())
_faiss = FaissIndexManager(
    settings.faiss_dir,
    use_gpu=settings.use_faiss_gpu,
    gpu_device=settings.faiss_gpu_device,
)


class RetrievalDebugRequest(BaseModel):
    query: str
    bm25_k: int = 20
    vec_k: int = 20
    top_k: int = 8
    bm25_mode: str = "heuristic"
    bm25_max_terms: int = 10
    rrf_k: int = 60
    w_bm25: float = 1.0
    w_vec: float = 1.0


async def _fetch_chunks_by_ids(
    db,
    ids: list[int],
) -> dict[int, dict[str, Any]]:
    """
    Fetch chunk rows for given ids, return as dict keyed by chunk_id.
    """
    if not ids:
        return {}

    placeholders = ",".join(["?"] * len(ids))
    sql = f"""
    SELECT
      c.id AS chunk_id,
      c.text AS text,
      c.metadata_json AS metadata_json,
      c.chunk_index AS chunk_index,
      c.document_id AS document_id,
      d.filename AS filename
    FROM chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE c.id IN ({placeholders});
    """

    async with db.execute(sql, tuple(ids)) as cursor:
        rows = await cursor.fetchall()

    out: dict[int, dict[str, Any]] = {}
    for r in rows:
        md = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
        cid = int(r["chunk_id"])
        out[cid] = {
            "chunk_id": cid,
            "text": r["text"],
            "filename": r["filename"],
            "metadata": md,
            "chunk_index": r["chunk_index"],
            "document_id": r["document_id"],
        }
    return out


async def _get_db_stats(db) -> dict[str, Any]:
    """Get database statistics for context."""
    stats = {}
    try:
        async with db.execute("SELECT COUNT(*) FROM documents") as cur:
            row = await cur.fetchone()
            stats["total_documents"] = row[0] if row else 0
        async with db.execute("SELECT COUNT(*) FROM chunks") as cur:
            row = await cur.fetchone()
            stats["total_chunks"] = row[0] if row else 0
        async with db.execute("SELECT COUNT(*) FROM chunks_fts") as cur:
            row = await cur.fetchone()
            stats["total_fts_entries"] = row[0] if row else 0
    except Exception as e:
        stats["error"] = str(e)
    return stats


@router.post("/debug/retrieval")
async def debug_retrieval(req: RetrievalDebugRequest):
    """
    Returns detailed retrieval internals for debugging hybrid RAG.

    Provides:
    - Query analysis (tokenization, BM25 query construction)
    - BM25 sparse retrieval results with scores
    - Vector dense retrieval results with similarity scores
    - RRF fusion details showing how results were combined
    - Timing information for each stage
    - Database statistics for context
    """
    timings: dict[str, float] = {}

    async with connect(settings.sqlite_path) as db:
        # Get database stats
        t0 = time.perf_counter()
        db_stats = await _get_db_stats(db)
        timings["db_stats_ms"] = (time.perf_counter() - t0) * 1000

        # Query analysis
        t0 = time.perf_counter()
        raw_tokens = req.query.lower().split()
        bm25_query_str = make_bm25_query(
            req.query, mode=req.bm25_mode, max_terms=req.bm25_max_terms
        )
        bm25_tokens = bm25_query_str.split() if bm25_query_str else []
        timings["query_analysis_ms"] = (time.perf_counter() - t0) * 1000

        query_analysis = {
            "original_query": req.query,
            "original_tokens": raw_tokens,
            "original_token_count": len(raw_tokens),
            "bm25_mode": req.bm25_mode,
            "bm25_max_terms": req.bm25_max_terms,
            "bm25_query": bm25_query_str,
            "bm25_tokens": bm25_tokens,
            "bm25_token_count": len(bm25_tokens),
            "tokens_removed": len(raw_tokens) - len(bm25_tokens),
        }

        # BM25 Search
        t0 = time.perf_counter()
        bm25_results = await bm25_search(
            db,
            req.query,
            k=req.bm25_k,
            mode=req.bm25_mode,
            max_terms=req.bm25_max_terms,
        )
        timings["bm25_search_ms"] = (time.perf_counter() - t0) * 1000

        # Add rank to BM25 results
        for i, r in enumerate(bm25_results):
            r["bm25_rank"] = i + 1

        # Vector Search
        t0 = time.perf_counter()
        qvec = np.array(_embedder.embed_query(req.query), dtype=np.float32)
        query_dim = int(qvec.shape[0])
        timings["embedding_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        if _faiss.cpu_index is None:
            _faiss.load_or_create(dim=query_dim)

        # Check for dimension mismatch (happens when embedding model changed)
        index_dim = _faiss.dim
        dimension_mismatch = index_dim is not None and query_dim != index_dim

        vec_results: list[dict[str, Any]] = []
        vec_error: str | None = None

        if dimension_mismatch:
            vec_error = (
                f"Dimension mismatch: Query embedding has {query_dim} dimensions, "
                f"but FAISS index expects {index_dim} dimensions. "
                f"This happens when the embedding model is changed after documents were ingested. "
                f"To fix: run /reset then /restart and re-ingest your documents with the new model."
            )
            timings["faiss_search_ms"] = 0
        else:
            try:
                vec = _faiss.search(qvec, k=req.vec_k)
                timings["faiss_search_ms"] = (time.perf_counter() - t0) * 1000

                # Fetch chunk details for vector results
                t0 = time.perf_counter()
                chunk_map = await _fetch_chunks_by_ids(db, vec.ids)
                timings["chunk_fetch_ms"] = (time.perf_counter() - t0) * 1000

                for rank, (cid, score) in enumerate(zip(vec.ids, vec.scores), start=1):
                    if cid in chunk_map:
                        item = dict(chunk_map[cid])
                        item["vec_score"] = float(score)
                        item["vec_rank"] = rank
                        vec_results.append(item)
            except ValueError as e:
                vec_error = str(e)
                timings["faiss_search_ms"] = 0

        # RRF Fusion
        t0 = time.perf_counter()
        fused = rrf_fuse(
            bm25_results=bm25_results,
            vec_results=vec_results,
            rrf_k=req.rrf_k,
            w_bm25=req.w_bm25,
            w_vec=req.w_vec,
            top_k=req.top_k,
        )
        timings["rrf_fusion_ms"] = (time.perf_counter() - t0) * 1000

        # Compute overlap analysis
        bm25_ids = {r["chunk_id"] for r in bm25_results}
        vec_ids = {r["chunk_id"] for r in vec_results}
        overlap_ids = bm25_ids & vec_ids
        bm25_only_ids = bm25_ids - vec_ids
        vec_only_ids = vec_ids - bm25_ids

        # Add source info to fused results
        for item in fused:
            cid = item["chunk_id"]
            item["in_bm25"] = cid in bm25_ids
            item["in_vector"] = cid in vec_ids
            item["in_both"] = cid in overlap_ids
            # Find ranks
            item["bm25_rank"] = next(
                (r["bm25_rank"] for r in bm25_results if r["chunk_id"] == cid), None
            )
            item["vec_rank"] = next(
                (r["vec_rank"] for r in vec_results if r["chunk_id"] == cid), None
            )

        # Compute RRF contribution breakdown for top results
        for item in fused:
            bm25_contrib = 0.0
            vec_contrib = 0.0
            if item["bm25_rank"]:
                bm25_contrib = req.w_bm25 / (req.rrf_k + item["bm25_rank"])
            if item["vec_rank"]:
                vec_contrib = req.w_vec / (req.rrf_k + item["vec_rank"])
            item["rrf_bm25_contribution"] = bm25_contrib
            item["rrf_vec_contribution"] = vec_contrib

        timings["total_ms"] = sum(timings.values())

    return {
        "query_analysis": query_analysis,
        "bm25": bm25_results,
        "vector": vec_results,
        "vector_error": vec_error,
        "fused": fused,
        "overlap_analysis": {
            "bm25_result_count": len(bm25_results),
            "vector_result_count": len(vec_results),
            "overlap_count": len(overlap_ids),
            "overlap_ids": sorted(overlap_ids),
            "bm25_only_count": len(bm25_only_ids),
            "bm25_only_ids": sorted(bm25_only_ids),
            "vector_only_count": len(vec_only_ids),
            "vector_only_ids": sorted(vec_only_ids)[:20],  # Limit for display
            "overlap_percentage": round(
                len(overlap_ids) / max(len(bm25_ids | vec_ids), 1) * 100, 1
            ),
        },
        "rrf_params": {
            "rrf_k": req.rrf_k,
            "w_bm25": req.w_bm25,
            "w_vec": req.w_vec,
            "top_k": req.top_k,
        },
        "db_stats": db_stats,
        "timings": timings,
        "debug": {
            "bm25_hits": len(bm25_results),
            "vec_hits": len(vec_results),
            "fused_hits": len(fused),
            "query_embedding_dim": query_dim,
            "faiss_index_dim": index_dim,
            "dimension_mismatch": dimension_mismatch,
            "faiss_ntotal": _faiss.cpu_index.ntotal if _faiss.cpu_index else 0,
        },
    }
