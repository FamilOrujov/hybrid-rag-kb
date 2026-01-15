from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter

from src.core.config import settings
from src.db.sqlite import connect
from src.api.routes_models import get_current_models

router = APIRouter()


async def _safe_count(db, table: str) -> tuple[int | None, str | None]:
    try:
        async with db.execute(f"SELECT COUNT(*) FROM {table};") as cur:
            row = await cur.fetchone()
        return (int(row[0]) if row else 0), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


@router.get("/stats")
async def stats() -> dict[str, Any]:
    """
    Portfolio endpoint.

    - SQLite: documents, chunks, chunks_fts counts
    - FAISS: index file existence, ntotal, dimension d, index type
    - GPU: whether FAISS GPU functions are available and how many GPUs FAISS sees
    - Repro: model names and chunking parameters (if present in settings)
    """

    # SQLite stats
    sqlite_path = Path(settings.sqlite_path)
    sqlite_info: dict[str, Any] = {
        "path": str(sqlite_path),
        "path_resolved": str(sqlite_path.resolve()),
        "documents": None,
        "chunks": None,
        "chunks_fts": None,
        "errors": {},
    }

    try:
        async with connect(settings.sqlite_path) as db:
            docs, docs_err = await _safe_count(db, "documents")
            chunks, chunks_err = await _safe_count(db, "chunks")
            fts, fts_err = await _safe_count(db, "chunks_fts")

        sqlite_info["documents"] = docs
        sqlite_info["chunks"] = chunks
        sqlite_info["chunks_fts"] = fts
        sqlite_info["errors"] = {
            "documents": docs_err,
            "chunks": chunks_err,
            "chunks_fts": fts_err,
        }
    except Exception as e:
        sqlite_info["errors"] = {"connect": f"{type(e).__name__}: {e}"}


    # FAISS stats (from disk)
    faiss_dir = Path(settings.faiss_dir)
    index_path = faiss_dir / "index.faiss"
    exists = index_path.exists()

    faiss_info: dict[str, Any] = {
        "dir": str(faiss_dir),
        "dir_resolved": str(faiss_dir.resolve()),
        "index_path": str(index_path),
        "exists": exists,
        "file_size_bytes": int(index_path.stat().st_size) if exists else 0,
        "ntotal": 0,
        "d": None,
        "index_type": None,
        "is_trained": None,
        "error": None,
    }


    # FAISS GPU capability (availability, not "currently loaded index")
    gpu_info: dict[str, Any] = {
        "configured_use_gpu": bool(getattr(settings, "use_faiss_gpu", False)),
        "configured_gpu_device": int(getattr(settings, "faiss_gpu_device", 0)),
        "faiss_gpu_build": False,
        "gpu_count_visible_to_faiss": 0,
        "note": (
            "FAISS disk I/O stores CPU indexes. A CPU index is typically cloned to GPU at runtime."
        ),
        "error": None,
    }

    try:
        import faiss  # provided by faiss-gpu or faiss-cpu

        # GPU build detection. If StandardGpuResources exists, this is a GPU-capable build
        gpu_info["faiss_gpu_build"] = hasattr(faiss, "StandardGpuResources")

        if hasattr(faiss, "get_num_gpus"):
            gpu_info["gpu_count_visible_to_faiss"] = int(faiss.get_num_gpus())

        if exists:
            idx = faiss.read_index(str(index_path))
            faiss_info["ntotal"] = int(idx.ntotal)  # number of vectors indexed
            faiss_info["d"] = int(getattr(idx, "d", 0)) if getattr(idx, "d", None) is not None else None
            faiss_info["index_type"] = type(idx).__name__
            faiss_info["is_trained"] = bool(getattr(idx, "is_trained", True))
    except Exception as e:
        # Could be FAISS import error, read_index error, etc
        faiss_info["ntotal"] = None
        faiss_info["error"] = f"{type(e).__name__}: {e}"
        gpu_info["error"] = f"{type(e).__name__}: {e}"


    # Reproducibility info (use runtime model state, not settings)
    current_models = get_current_models()
    repro: dict[str, Any] = {
        "ollama_base_url": settings.ollama_base_url,
        "chat_model": current_models["chat_model"],
        "embed_model": current_models["embed_model"],
        "num_predict": int(getattr(settings, "ollama_num_predict", 0)),
        # Chunking params. If you have them in settings, they will appear here
        "chunk_size": getattr(settings, "chunk_size", None),
        "chunk_overlap": getattr(settings, "chunk_overlap", None),
    }

    return {
        "sqlite": sqlite_info,
        "faiss": faiss_info,
        "gpu": gpu_info,
        "repro": repro,
    }

