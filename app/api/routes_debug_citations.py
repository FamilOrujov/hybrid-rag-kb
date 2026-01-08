from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel
from langchain_ollama import ChatOllama

from app.core.config import settings
from app.db.sqlite import connect
from app.rag.embeddings import make_embedder
from app.rag.vectorstore import FaissIndexManager
from app.rag.qa import answer_question
from app.rag.citations import validate_citations_detailed

router = APIRouter()


_embedder = make_embedder(settings.ollama_base_url, settings.ollama_embed_model)

_llm = ChatOllama(
    model=settings.ollama_chat_model,
    temperature=0,
    base_url=settings.ollama_base_url,
    num_predict=settings.ollama_num_predict,  # max tokens generated :contentReference[oaicite:1]{index=1}
)

_faiss = FaissIndexManager(
    settings.faiss_dir,
    use_gpu=settings.use_faiss_gpu,
    gpu_device=settings.faiss_gpu_device,
)


class DebugCitationsRequest(BaseModel):
    query: str
    session_id: str = "debug"

    bm25_k: int = 20
    vec_k: int = 20
    top_k: int = 8

    # BM25 controls.
    # bm25_mode:
    #   - "raw": use raw user query tokens (safe tokenization only)
    #   - "heuristic": keyword filtered query (stopwords removed, short tokens removed, top-N)
    bm25_mode: str = "heuristic"
    # If set, this overrides bm25_mode and is used directly as the FTS5 MATCH query string.
    bm25_query: str | None = None
    bm25_max_terms: int = 10

    min_unique_citations: int = 1
    require_citation_per_paragraph: bool = True


@router.post("/debug/citations")
async def debug_citations(req: DebugCitationsRequest) -> dict[str, Any]:
    """
    Run the same pipeline as /query, then return citation diagnostics.
    """
    async with connect(settings.sqlite_path) as db:
        result = await answer_question(
            db=db,
            session_id=req.session_id,
            query=req.query,
            chat_model=_llm,
            embedder=_embedder,
            faiss_mgr=_faiss,
            bm25_k=req.bm25_k,
            vec_k=req.vec_k,
            final_k=req.top_k,
            bm25_mode=req.bm25_mode,
            bm25_query=req.bm25_query,
            bm25_max_terms=req.bm25_max_terms,
            memory_k=0,
            store_memory=False,
            min_unique_citations=req.min_unique_citations,
            require_citation_per_paragraph=req.require_citation_per_paragraph,
            rewrite_on_fail=True,
        )

    answer = result.get("answer", "")
    sources = result.get("sources", [])
    allowed = [int(s["chunk_id"]) for s in sources if "chunk_id" in s]

    ok, report = validate_citations_detailed(
        answer_text=answer,
        allowed_chunk_ids=allowed,
        min_unique_citations=req.min_unique_citations,
        require_citation_per_paragraph=req.require_citation_per_paragraph,
    )

    return {
        "ok": ok,
        "report": report,
        "answer": answer,
        "allowed_chunk_ids": allowed,
        "sources": sources,
        "retrieval_debug": result.get("debug", {}),
    }
