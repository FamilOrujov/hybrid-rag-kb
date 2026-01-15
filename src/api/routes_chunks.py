from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException

from src.core.config import settings
from src.db.sqlite import connect

router = APIRouter()



@router.get("/chunks/{chunk_id}")
async def get_chunk(chunk_id: int) -> dict[str, Any]:
    """
    Fetch one chunk by its SQLite primary key (chunks.id).

    Why this endpoint exists:
    - debug endpoints and /query sources return chunk_id.
    - This lets you inspect the exact text and metadata that was fed to the model.

    FastAPI path params:
    - /chunks/{chunk_id} captures chunk_id and passes it to the function. :contentReference[oaicite:5]{index=5}
    """
    sql = """
    SELECT
      c.id AS chunk_id,
      c.text AS text,
      c.metadata_json AS metadata_json,
      c.document_id AS document_id,
      c.chunk_index AS chunk_index,
      d.filename AS filename
    FROM chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE c.id = ?;
    """

    async with connect(settings.sqlite_path) as db:
        async with db.execute(sql, (chunk_id,)) as cursor:
            row = await cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail=f"chunk_id {chunk_id} not found")

    metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}

    return {
        "chunk_id": int(row["chunk_id"]),
        "document_id": int(row["document_id"]),
        "filename": row["filename"],
        "chunk_index": int(row["chunk_index"]),
        "metadata": metadata,
        "text": row["text"],
    }

