from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from src.api.model_config import get_initial_embed_model
from src.core.config import settings
from src.db.sqlite import connect
from src.rag.embeddings import make_embedder
from src.rag.ingest import ingest_files
from src.rag.vectorstore import FaissIndexManager

router = APIRouter()

_embedder = make_embedder(settings.ollama_base_url, get_initial_embed_model())
_faiss = FaissIndexManager(
    settings.faiss_dir, use_gpu=settings.use_faiss_gpu, gpu_device=settings.faiss_gpu_device
)


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
