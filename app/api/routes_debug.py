from __future__ import annotations

import json
from typing import Any

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import settings
from app.db.sqlite import connect
from app.rag.bm25_fts import bm25_search
from app.rag.embeddings import make_embedder
from app.rag.hybrid_fusion import rrf_fuse
from app.rag.vectorstore import FaissIndexManager

router = APIRouter()

_embedder = make_embedder(settings.ollama_base_url, settings.ollama_embed_model)
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


async def _fetch_chunks_by_ids(
    db,
    ids: list[int],
) -> dict[int, dict[str, Any]]:
    """
    Fetch chunk rows for given ids, return as dict keyed by chunk_id.
    This lets us reassemble vector results in the exact ranked order from FAISS.
    """
    if not ids:
        return {}

    placeholders = ",".join(["?"] * len(ids))
    sql = f"""
    SELECT
      c.id AS chunk_id,
      c.text AS text,
      c.metadata_json AS metadata_json,
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
        }
    return out


@router.post("/debug/retrieval")
async def debug_retrieval(req: RetrievalDebugRequest):
    """
    Returns retrieval internals so you can prove hybrid retrieval is working.

    - bm25. SQLite FTS5 MATCH ordered by bm25(). Smaller is better. :contentReference[oaicite:1]{index=1}
    - vector. FAISS similarity hits.
    - fused. RRF merge of both ranked lists.
    """
    async with connect(settings.sqlite_path) as db:
        bm25_results = await bm25_search(db, req.query, k=req.bm25_k)

        qvec = np.array(_embedder.embed_query(req.query), dtype=np.float32)
        if _faiss.cpu_index is None:
            _faiss.load_or_create(dim=int(qvec.shape[0]))

        vec = _faiss.search(qvec, k=req.vec_k)

        chunk_map = await _fetch_chunks_by_ids(db, vec.ids)
        vec_results: list[dict[str, Any]] = []
        for cid, score in zip(vec.ids, vec.scores):
            if cid in chunk_map:
                item = dict(chunk_map[cid])
                item["vec_score"] = float(score)
                vec_results.append(item)

        fused = rrf_fuse(
            bm25_results=bm25_results,
            vec_results=vec_results,
            top_k=req.top_k,
        )

    return {
        "bm25": bm25_results,
        "vector": vec_results,
        "fused": fused,
        "debug": {
            "bm25_hits": len(bm25_results),
            "vec_hits": len(vec_results),
            "fused_hits": len(fused),
        },
    }
