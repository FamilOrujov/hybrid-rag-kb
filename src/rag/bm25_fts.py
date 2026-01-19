from __future__ import annotations

import json
import re
from typing import Any

import aiosqlite

# Small practical stopword set
_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "not",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "by",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "as",
    "at",
    "it",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "we",
    "they",
    "he",
    "she",
    "my",
    "your",
    "our",
    "their",
    "summarize",
    "summary",
    "main",
    "points",
    "cite",
    "sources",
    "document",
    "documents",
    "uploaded",
}

_WORD_RE = re.compile(r"\w+")


def make_bm25_query(
    user_query: str,
    *,
    mode: str = "heuristic",
    max_terms: int = 10,
) -> str:
    """
    Build a query string for SQLite FTS5 MATCH.

    Why.
    - FTS5 MATCH parses a query language (operators, phrases, NEAR, etc). :contentReference[oaicite:2]{index=2}
    - Passing raw punctuation can cause syntax errors, so we tokenize safely.
    - Some user queries are instruction-like ("summarize...") and do not overlap
      with document text, giving bm25_hits=0.

    Modes.
    - raw: keep all tokens (after safe tokenization), no stopword filtering.
    - heuristic: remove stopwords + short tokens and keep top-N.

    Note.
    - If you want true FTS5 syntax (NEAR, quotes, column:term), pass bm25_query
      directly from the API instead of using this helper.
    """
    tokens = [t.lower() for t in _WORD_RE.findall(user_query)]

    if mode == "raw":
        return " ".join(tokens).strip()

    # heuristic
    seen: set[str] = set()
    kept: list[str] = []
    for t in tokens:
        if len(t) < 3:
            continue
        if t in _STOPWORDS:
            continue
        if t in seen:
            continue
        seen.add(t)
        kept.append(t)
        if len(kept) >= max_terms:
            break

    return " ".join(kept).strip()


async def bm25_search(
    db: aiosqlite.Connection,
    user_query: str,
    *,
    k: int = 20,
    mode: str = "heuristic",
    bm25_query: str | None = None,
    max_terms: int = 10,
) -> list[dict[str, Any]]:
    """
    Sparse retrieval using SQLite FTS5.

    - Uses: WHERE chunks_fts MATCH ?
    - Ranks with: bm25(chunks_fts)
    - Typical usage orders by bm25() to get best matches first. :contentReference[oaicite:3]{index=3}
    """
    q = (bm25_query or make_bm25_query(user_query, mode=mode, max_terms=max_terms)).strip()
    if not q:
        return []

    sql = """
    SELECT
      c.id             AS chunk_id,
      c.text           AS text,
      c.metadata_json  AS metadata_json,
      d.filename       AS filename,
      bm25(chunks_fts) AS bm25_score
    FROM chunks_fts
    JOIN chunks c    ON chunks_fts.rowid = c.id
    JOIN documents d ON c.document_id = d.id
    WHERE chunks_fts MATCH ?
    ORDER BY bm25(chunks_fts)
    LIMIT ?;
    """

    async with db.execute(sql, (q, k)) as cursor:
        rows = await cursor.fetchall()

    out: list[dict[str, Any]] = []
    for r in rows:
        md = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
        out.append(
            {
                "chunk_id": int(r["chunk_id"]),
                "text": r["text"],
                "metadata": md,
                "filename": r["filename"],
                "bm25_score": float(r["bm25_score"]),
            }
        )
    return out
