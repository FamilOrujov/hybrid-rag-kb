from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from app.core.config import settings
from app.db.sqlite import connect
from app.rag.embeddings import make_embedder
from app.rag.vectorstore import FaissIndexManager
from app.rag.ingest import ingest_files

router = APIRouter()

_embedder = make_embedder(settings.ollama_base_url, settings.ollama_embed_model)
_faiss = FaissIndexManager(settings.faiss_dir, use_gpu=settings.use_faiss_gpu, gpu_device=settings.faiss_gpu_device)

@router.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    received = [f.filename for f in files]
    async with connect(settings.sqlite_path) as db:
        result = await ingest_files(
            db=db,
            files=files,
            raw_dir=settings.raw_dir,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            embedder=_embedder,
            faiss_mgr=_faiss,
        )
    return {"received": received, **result}
