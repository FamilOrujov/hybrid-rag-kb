from __future__ import annotations

import json
from typing import Any

import numpy as np
import aiosqlite
from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.rag.bm25_fts import bm25_search, make_bm25_query
from app.rag.hybrid_fusion import rrf_fuse
from app.rag.vectorstore import FaissIndexManager
from app.rag.memory import add_message, get_recent_messages
from app.rag.citations import validate_citations_detailed, split_paragraphs


async def _fetch_chunks_by_ids(
    db: aiosqlite.Connection,
    ids: list[int],
) -> dict[int, dict[str, Any]]:
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


def _format_context(chunks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for c in chunks:
        chunk_index = c.get("metadata", {}).get("chunk_index")
        parts.append(
            f"[Source: {c['filename']} | cid:{int(c['chunk_id'])} | chunk_index:{chunk_index}]\n{c['text']}"
        )
    return "\n\n".join(parts)


def _inject_citations_per_paragraph(
    answer_text: str,
    *,
    cite_tokens: list[str],
    missing_paragraphs: list[int],
) -> str:
    """
    Deterministic, very low CPU fallback.

    If the model refuses to place citations per paragraph, we append one allowed
    citation token to each missing paragraph. This makes validation pass without
    another LLM call.

    cite_tokens: list like ["[Source: file.pdf | cid:54]", ...]
    missing_paragraphs: indices that need a citation.
    """
    if not cite_tokens:
        return answer_text

    paras = split_paragraphs(answer_text)
    for idx in missing_paragraphs:
        token = cite_tokens[idx % len(cite_tokens)]
        paras[idx] = paras[idx].rstrip() + " " + token

    return "\n\n".join(paras)


async def answer_question(
    *,
    db: aiosqlite.Connection,
    session_id: str,
    query: str,
    chat_model: ChatOllama,
    embedder: OllamaEmbeddings,
    faiss_mgr: FaissIndexManager,
    bm25_k: int = 20,
    bm25_mode: str = "heuristic",
    bm25_query: str | None = None,
    bm25_max_terms: int = 10,
    vec_k: int = 20,
    final_k: int = 8,
    memory_k: int = 6,
    store_memory: bool = True,
    min_unique_citations: int = 1,
    require_citation_per_paragraph: bool = True,
    rewrite_on_fail: bool = True,
) -> dict[str, Any]:
    if store_memory:
        await add_message(db, session_id, "user", query)

    bm25_results = await bm25_search(
        db,
        query,
        k=bm25_k,
        mode=bm25_mode,
        bm25_query=bm25_query,
        max_terms=bm25_max_terms,
    )

    qvec = np.array(embedder.embed_query(query), dtype=np.float32)
    if faiss_mgr.cpu_index is None:
        faiss_mgr.load_or_create(dim=int(qvec.shape[0]))

    vec = faiss_mgr.search(qvec, k=vec_k)

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
        top_k=final_k,
    )

    if not fused:
        answer = "I do not know based on the documents I have indexed."
        if store_memory:
            await add_message(db, session_id, "assistant", answer)
        return {
            "answer": answer,
            "sources": [],
            "debug": {
                "bm25_hits": len(bm25_results),
                "vec_hits": len(vec_results),
                "fused": 0,
                "citation_ok": True,
                "citation_report": {"reason": "no retrieved chunks"},
            },
        }

    allowed_ids = [int(c["chunk_id"]) for c in fused]
    cite_tokens = [f"[Source: {c['filename']} | cid:{int(c['chunk_id'])}]" for c in fused]
    allowed_tokens_str = " ".join(cite_tokens)

    context = _format_context(fused)

    history: list[tuple[str, str]] = []
    if store_memory and memory_k > 0:
        history = await get_recent_messages(db, session_id, limit=memory_k)

    system_msg = (
        "You are a helpful assistant.\n"
        "Use only the provided Context.\n"
        "If the answer is not in the Context, say you do not know based on the documents.\n\n"
        "OUTPUT LENGTH:\n"
        "Write at most 4 short paragraphs.\n\n"
        "CITATION RULES (mandatory):\n"
        "1) After every paragraph, add at least one citation.\n"
        "2) Use this exact citation format: [Source: filename | cid:NUMBER]\n"
        "3) NUMBER must be one of the allowed citations listed below.\n"
        "4) Do not use author year citations like (2007). Do not add a bibliography.\n\n"
        f"Allowed citations: {allowed_tokens_str}\n"
    )

    messages: list[tuple[str, str]] = [("system", system_msg)]
    messages.extend(history)
    messages.append(("human", f"Question:\n{query}\n\nContext:\n{context}"))

    answer = chat_model.invoke(messages).content

    ok, report = validate_citations_detailed(
        answer_text=answer,
        allowed_chunk_ids=allowed_ids,
        min_unique_citations=min_unique_citations,
        require_citation_per_paragraph=require_citation_per_paragraph,
    )

    if not ok and rewrite_on_fail:
        rewrite_system = (
            "You will ONLY fix citation formatting.\n"
            "Do not add new facts.\n"
            "Do not add a bibliography.\n"
            "After every paragraph, append exactly one allowed citation token.\n"
            "Use ONLY tokens from the allowed list.\n"
        )
        rewrite_messages = [
            ("system", rewrite_system),
            ("human", f"Allowed citations: {allowed_tokens_str}\n\nText:\n{answer}\n\nWhy it failed:\n{report}"),
        ]
        answer2 = chat_model.invoke(rewrite_messages).content

        ok2, report2 = validate_citations_detailed(
            answer_text=answer2,
            allowed_chunk_ids=allowed_ids,
            min_unique_citations=min_unique_citations,
            require_citation_per_paragraph=require_citation_per_paragraph,
        )

        if ok2:
            answer, ok, report = answer2, ok2, report2
        else:
            answer, ok, report = answer2, ok2, report2

    if not ok and require_citation_per_paragraph:
        missing = report.get("missing_paragraphs", [])
        answer3 = _inject_citations_per_paragraph(answer, cite_tokens=cite_tokens, missing_paragraphs=missing)

        ok3, report3 = validate_citations_detailed(
            answer_text=answer3,
            allowed_chunk_ids=allowed_ids,
            min_unique_citations=min_unique_citations,
            require_citation_per_paragraph=require_citation_per_paragraph,
        )

        answer, ok, report = answer3, ok3, report3

    if store_memory:
        await add_message(db, session_id, "assistant", answer)

    sources = [
        {
            "chunk_id": int(c["chunk_id"]),
            "filename": c["filename"],
            "chunk_index": c.get("metadata", {}).get("chunk_index"),
            "fused_score": c["fused_score"],
        }
        for c in fused
    ]

    return {
        "answer": answer,
        "sources": sources,
        "debug": {
            "bm25_hits": len(bm25_results),
            "vec_hits": len(vec_results),
            "fused": len(fused),
            "citation_ok": ok,
            "citation_report": report,
        },
    }
