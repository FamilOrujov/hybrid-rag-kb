from __future__ import annotations

import re
import json
from typing import Any

import numpy as np
import aiosqlite
from langchain_ollama import ChatOllama, OllamaEmbeddings

from src.rag.bm25_fts import bm25_search, make_bm25_query
from src.rag.hybrid_fusion import rrf_fuse
from src.rag.vectorstore import FaissIndexManager
from src.rag.memory import add_message, get_recent_messages
from src.rag.citations import validate_citations_detailed, split_paragraphs


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
    """Format chunks for context, with clear source markers."""
    parts: list[str] = []
    for c in chunks:
        cid = int(c["chunk_id"])
        filename = c["filename"]
        text = c["text"].strip()
        parts.append(f"[cid:{cid}] from {filename}:\n{text}")
    return "\n\n---\n\n".join(parts)


def _clean_answer(answer: str, allowed_ids: set[int]) -> str:
    """
    Post-process the answer to remove common LLM artifacts.
    
    Removes:
    - Preamble like "Okay, here's...", "Here is the answer...", etc.
    - Meta-commentary about formatting or citations
    - Trailing bibliography sections
    """
    # Common preamble patterns to remove
    preamble_patterns = [
        r"^(?:Okay|OK|Sure|Certainly|Of course)[,.]?\s*(?:here'?s?|I'?ll|let me)[^.]*[.!]\s*",
        r"^(?:Here is|Here's|Below is)[^.]*[.!:]\s*",
        r"^(?:Based on|According to) (?:the )?(?:provided |given )?(?:context|documents?|sources?)[,.]?\s*",
        r"^(?:The )?(?:corrected |revised |formatted )?(?:text|answer|response)[^.]*[.:]\s*",
        r"^I (?:understand|see)[^.]*[.!]\s*",
    ]
    
    cleaned = answer.strip()
    for pattern in preamble_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove trailing bibliography/references sections
    bib_patterns = [
        r"\n+(?:References|Bibliography|Sources|Works Cited):?\s*\n.*$",
        r"\n+D'Mello.*$",  # Common incomplete citation
        r"\n+\[\d+\][^\[]*$",  # Numbered reference at end
    ]
    for pattern in bib_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove lines that are just author names/affiliations (common artifact)
    lines = cleaned.split("\n")
    filtered_lines = []
    for line in lines:
        line_stripped = line.strip()
        # Skip lines that look like author affiliations or emails
        if re.match(r"^[\d\s]*Department of", line_stripped, re.IGNORECASE):
            continue
        if re.match(r"^[\w\s,]+@[\w.]+$", line_stripped):  # Email pattern
            continue
        if re.match(r"^[A-Z][a-z]+\s+[A-Z][a-z]+\d*$", line_stripped):  # "Name Name1" pattern
            continue
        if re.match(r"^(?:Viale|Via|Street|Avenue)\s", line_stripped, re.IGNORECASE):
            continue
        filtered_lines.append(line)
    
    cleaned = "\n".join(filtered_lines).strip()
    
    # Clean up multiple blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    
    return cleaned.strip()


def _inject_citations_per_paragraph(
    answer_text: str,
    *,
    cite_tokens: list[str],
    missing_paragraphs: list[int],
) -> str:
    """
    Deterministic fallback: append citations to paragraphs missing them.
    """
    if not cite_tokens:
        return answer_text

    paras = split_paragraphs(answer_text)
    for idx in missing_paragraphs:
        if idx < len(paras):
            token = cite_tokens[idx % len(cite_tokens)]
            paras[idx] = paras[idx].rstrip() + " " + token

    return "\n\n".join(paras)


def _build_system_prompt(allowed_ids: list[int]) -> str:
    """Build a clear, direct system prompt."""
    cid_list = ", ".join(str(cid) for cid in allowed_ids)
    
    return f"""You are a research assistant. Your task is to answer questions using ONLY the provided source documents.

RESPONSE FORMAT:
- Write 2 to 3 concise paragraphs that directly answer the question
- End each paragraph with a citation: [Source: filename | cid:NUMBER]
- Use ONLY these citation IDs: {cid_list}

STRICT RULES:
- Start your answer immediately with the content. No introductions.
- Do NOT write phrases like "Here's the answer" or "Based on the context" or "Okay, here's"
- Do NOT copy author names, email addresses, or institutional affiliations
- Do NOT include bibliography entries or reference lists
- Do NOT include movie quotes or unrelated content
- SYNTHESIZE information in your own words, do not copy chunks verbatim
- If you cannot answer from the sources, say "I don't have enough information to answer this question."

Your response should read like a well-written encyclopedia entry, not a collection of copied text."""


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

    # Hybrid retrieval: BM25 + Vector search
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

    # Reciprocal Rank Fusion
    fused = rrf_fuse(
        bm25_results=bm25_results,
        vec_results=vec_results,
        top_k=final_k,
    )

    if not fused:
        answer = "I don't have enough information in the indexed documents to answer this question."
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
    allowed_set = set(allowed_ids)
    cite_tokens = [f"[Source: {c['filename']} | cid:{int(c['chunk_id'])}]" for c in fused]

    context = _format_context(fused)
    system_msg = _build_system_prompt(allowed_ids)

    # Build message history
    history: list[tuple[str, str]] = []
    if store_memory and memory_k > 0:
        history = await get_recent_messages(db, session_id, limit=memory_k)

    messages: list[tuple[str, str]] = [("system", system_msg)]
    messages.extend(history)
    messages.append(("human", f"Question: {query}\n\nSource Documents:\n{context}"))

    # Generate answer
    answer = chat_model.invoke(messages).content
    
    # Post-process to clean common artifacts
    answer = _clean_answer(answer, allowed_set)

    # Validate citations
    ok, report = validate_citations_detailed(
        answer_text=answer,
        allowed_chunk_ids=allowed_ids,
        min_unique_citations=min_unique_citations,
        require_citation_per_paragraph=require_citation_per_paragraph,
    )

    # If validation fails, try to fix citations (without rewrite LLM call)
    if not ok:
        # First, try injecting missing citations
        missing = report.get("missing_paragraphs", [])
        if missing:
            answer = _inject_citations_per_paragraph(
                answer, 
                cite_tokens=cite_tokens, 
                missing_paragraphs=missing
            )
            ok, report = validate_citations_detailed(
                answer_text=answer,
                allowed_chunk_ids=allowed_ids,
                min_unique_citations=min_unique_citations,
                require_citation_per_paragraph=require_citation_per_paragraph,
            )
        
        # If still failing due to invalid IDs, try to fix them
        if not ok and report.get("invalid_ids"):
            # Replace invalid citation IDs with valid ones
            invalid_ids = report.get("invalid_ids", [])
            for invalid_id in invalid_ids:
                # Replace with the first valid ID
                if allowed_ids:
                    replacement = f"[Source: {fused[0]['filename']} | cid:{allowed_ids[0]}]"
                    answer = re.sub(
                        rf"\[Source:[^\]]*cid:{invalid_id}[^\]]*\]",
                        replacement,
                        answer
                    )
            
            ok, report = validate_citations_detailed(
                answer_text=answer,
                allowed_chunk_ids=allowed_ids,
                min_unique_citations=min_unique_citations,
                require_citation_per_paragraph=require_citation_per_paragraph,
            )

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
