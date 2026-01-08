from __future__ import annotations

from typing import Any


def rrf_fuse(
    bm25_results: list[dict[str, Any]],
    vec_results: list[dict[str, Any]],
    *,
    rrf_k:int = 60,
    w_bm25: float = 1.0,
    w_vec: float = 1.0,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """
    Reciprocal Rank Fusion.

    We fuse by rank, not by raw scores.
    That avoids the problem that BM25 and cosine similarity live on different scales.

    fused_score += weight / (rrf_k + rank)
    rank is 1-based.
    """
    fused: dict[int, dict[str, Any]] = {}

    for rank, item in enumerate(bm25_results, start=1):
        cid = int(item["chunk_id"])
        add = w_bm25 / (rrf_k + rank)
        if cid not in fused:
            fused[cid] = {
                "chunk_id": cid,
                "text": item["text"],
                "filename": item["filename"],
                "metadata": item.get("metadata", {}),
                "bm25_score": item.get("bm25_score"),
                "vec_score": None,
                "fused_score": 0.0,
            }
        fused[cid]["fused_score"] += add

    for rank, item in enumerate(vec_results, start=1):
        cid = int(item["chunk_id"])
        add = w_vec / (rrf_k + rank)
        if cid not in fused:
            fused[cid] = {
                "chunk_id": cid,
                "text": item["text"],
                "filename": item["filename"],
                "metadata": item.get("metadata", {}),
                "bm25_score": None,
                "vec_score": item.get("vec_score"),
                "fused_score": 0.0,
            }
        else:
            fused[cid]["vec_score"] = item.get("vec_score")
        fused[cid]["fused_score"] += add

    
    merged = list(fused.values())
    merged.sort(key=lambda x: x["fused_score"], reverse=True)
    return merged[:top_k]

