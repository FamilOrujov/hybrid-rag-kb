from __future__ import annotations

from langchain_ollama import OllamaEmbeddings


def make_embedder(base_url: str, model: str) -> OllamaEmbeddings:
    return OllamaEmbeddings(base_url=base_url, model=model)

