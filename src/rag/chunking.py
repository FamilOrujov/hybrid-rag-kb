from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any]


def chunk_text(
    text: str, base_metadata: dict[str, Any], chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    parts = splitter.split_text(text)
    out: list[Chunk] = []
    for idx, part in enumerate(parts):
        md = dict(base_metadata)
        md["chunk_index"] = idx
        out.append(Chunk(text=part, metadata=md))
    return out
