from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional

import aiosqlite
import numpy as np
from fastapi import UploadFile

from src.rag.loaders import load_text_from_path
from src.rag.chunking import chunk_text
from src.rag.vectorstore import FaissIndexManager


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def _fetchone(db: aiosqlite.Connection, sql: str, params: tuple) -> Optional[aiosqlite.Row]:
    """
    aiosqlite does not expose execute_fetchone() in its documented API.
    The standard pattern is execute() then cursor.fetchone(). :contentReference[oaicite:2]{index=2}
    """
    async with db.execute(sql, params) as cursor:
        return await cursor.fetchone()


async def ingest_files(
    db: aiosqlite.Connection,
    files: List[UploadFile],
    raw_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    embedder,
    faiss_mgr: FaissIndexManager,
) -> dict:
    Path(raw_dir).mkdir(parents=True, exist_ok=True)

    ingested_docs = 0
    ingested_chunks = 0
    ingested_vectors = 0

    for f in files:
        data = await f.read()
        digest = _sha256_bytes(data)

        # Dedup by sha256
        row = await _fetchone(db, "SELECT id FROM documents WHERE sha256 = ?", (digest,))
        if row is not None:
            continue

        safe_name = f.filename or "upload.bin"
        stored_path = str(Path(raw_dir) / f"{digest}_{safe_name}")
        Path(stored_path).write_bytes(data)

        cur = await db.execute(
            "INSERT INTO documents(filename, sha256, content_type, stored_path) VALUES(?,?,?,?)",
            (safe_name, digest, f.content_type, stored_path),
        )
        doc_id = int(cur.lastrowid)

        text, loader_meta = load_text_from_path(stored_path)
        base_meta = {"document_id": doc_id, "filename": safe_name, **loader_meta}

        chunks = chunk_text(text, base_meta, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        chunk_ids: list[int] = []
        chunk_texts: list[str] = []

        for ch in chunks:
            md_json = json.dumps(ch.metadata, ensure_ascii=False)
            cur2 = await db.execute(
                "INSERT INTO chunks(document_id, chunk_index, text, metadata_json) VALUES(?,?,?,?)",
                (doc_id, int(ch.metadata["chunk_index"]), ch.text, md_json),
            )
            chunk_id = int(cur2.lastrowid)
            chunk_ids.append(chunk_id)
            chunk_texts.append(ch.text)

        await db.commit()

        # Embed all chunk texts with Ollama embeddings
        vectors = embedder.embed_documents(chunk_texts)
        vecs = np.array(vectors, dtype=np.float32)

        if faiss_mgr.dim is None:
            faiss_mgr.load_or_create(dim=vecs.shape[1])

        faiss_mgr.add(chunk_ids, vecs)

        ingested_docs += 1
        ingested_chunks += len(chunks)
        ingested_vectors += len(chunk_ids)

    return {"documents_added": ingested_docs, "chunks_added": ingested_chunks, "vectors_added": ingested_vectors}
