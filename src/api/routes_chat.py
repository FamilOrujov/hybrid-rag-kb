from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from langchain_ollama import ChatOllama

from src.core.config import settings
from src.db.sqlite import connect
from src.rag.embeddings import make_embedder
from src.rag.vectorstore import FaissIndexManager
from src.rag.qa import answer_question
from src.api.model_config import get_initial_chat_model, get_initial_embed_model

router = APIRouter()

_embedder = make_embedder(settings.ollama_base_url, get_initial_embed_model())

_llm = ChatOllama(
    model=get_initial_chat_model(),
    base_url = settings.ollama_base_url,
    temperature=0,
    validate_model_on_init=True,
    num_predict=settings.ollama_num_predict,
)


_faiss = FaissIndexManager(
    settings.faiss_dir,
    use_gpu=settings.use_faiss_gpu,
    gpu_device=settings.faiss_gpu_device,
)


class QueryRequest(BaseModel):
    session_id: str
    query: str
    bm25_k: int = 20
    vec_k: int = 20
    top_k: int = 8
    memory_k: int = 6


@router.post("/query")
async def query(req: QueryRequest):
    async with connect(settings.sqlite_path) as db:
        result = await answer_question(
            db=db,
            session_id=req.session_id,
            query=req.query,
            chat_model=_llm,
            embedder=_embedder,
            faiss_mgr = _faiss,
            bm25_k=req.bm25_k,
            vec_k=req.vec_k,
            final_k=req.top_k,
            memory_k=req.memory_k,
        )
    return result
