from __future__ import annotations

import httpx
from langchain_ollama import OllamaEmbeddings


def make_embedder(base_url: str, model: str, timeout: float = 600.0) -> OllamaEmbeddings:
    return OllamaEmbeddings(
        base_url=base_url,
        model=model,
        client_kwargs={"timeout": httpx.Timeout(timeout)},
    )

