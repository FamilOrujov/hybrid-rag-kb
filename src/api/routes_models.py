"""Model management API routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from src.api.model_config import get_initial_chat_model, get_initial_embed_model
from src.core.config import settings
from src.rag.embeddings import make_embedder

router = APIRouter()


class ModelUpdateRequest(BaseModel):
    """Request to update active models."""
    chat_model: str | None = None
    embed_model: str | None = None


# Persistent configuration file path (stored alongside other data)
_MODEL_CONFIG_PATH = Path(settings.faiss_dir).parent.parent / "model_config.json"


def _load_persistent_config() -> dict[str, str]:
    """Load model configuration from persistent storage.

    Returns config from file if it exists, otherwise returns empty dict.
    The file stores user's model preferences that persist across restarts.
    """
    if _MODEL_CONFIG_PATH.exists():
        try:
            with open(_MODEL_CONFIG_PATH) as f:
                config = json.load(f)
                return {
                    "chat_model": config.get("chat_model"),
                    "embed_model": config.get("embed_model"),
                }
        except (OSError, json.JSONDecodeError):
            pass  # Fall back to defaults on error
    return {}


def _save_persistent_config(chat_model: str | None = None, embed_model: str | None = None) -> None:
    """Save model configuration to persistent storage.

    Only saves non-None values. Preserves existing values for keys not provided.
    """
    # Ensure parent directory exists
    _MODEL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load existing config to preserve values not being updated
    existing = _load_persistent_config()

    new_config = {
        "chat_model": chat_model if chat_model is not None else existing.get("chat_model") or settings.ollama_chat_model,
        "embed_model": embed_model if embed_model is not None else existing.get("embed_model") or settings.ollama_embed_model,
    }

    try:
        with open(_MODEL_CONFIG_PATH, "w") as f:
            json.dump(new_config, f, indent=2)
    except OSError:
        pass  # Silently fail if we can't write (read-only filesystem, etc.)


# Runtime model state - initialized from persistent config if available
_current_chat_model: str = get_initial_chat_model()
_current_embed_model: str = get_initial_embed_model()


def get_current_models() -> dict[str, str]:
    """Get currently active models."""
    return {
        "chat_model": _current_chat_model,
        "embed_model": _current_embed_model,
    }


@router.get("/models")
async def list_models() -> dict[str, Any]:
    """
    List available Ollama models and current configuration.

    Queries the local Ollama server for available models.
    """
    global _current_chat_model, _current_embed_model

    ollama_models: list[dict[str, Any]] = []
    error: str | None = None

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                ollama_models = data.get("models", [])
            else:
                error = f"Ollama API returned {response.status_code}"
    except httpx.ConnectError:
        error = "Cannot connect to Ollama. Is it running?"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    # Categorize models
    chat_models: list[dict[str, Any]] = []
    embed_models: list[dict[str, Any]] = []

    for model in ollama_models:
        name = model.get("name", "")
        size = model.get("size", 0)
        modified = model.get("modified_at", "")

        model_info = {
            "name": name,
            "size_gb": round(size / (1024**3), 2) if size else 0,
            "modified": modified,
        }

        # Heuristic: embed models typically have "embed" in name
        if "embed" in name.lower():
            embed_models.append(model_info)
        else:
            chat_models.append(model_info)

    return {
        "current": {
            "chat_model": _current_chat_model,
            "embed_model": _current_embed_model,
            "ollama_base_url": settings.ollama_base_url,
        },
        "available": {
            "chat_models": chat_models,
            "embed_models": embed_models,
            "all_models": [m.get("name") for m in ollama_models],
        },
        "error": error,
    }


@router.post("/models")
async def update_models(req: ModelUpdateRequest) -> dict[str, Any]:
    """
    Update the active chat and/or embedding model.

    Changes are persisted to disk and will survive server restarts.
    The persistent config takes precedence over .env defaults.
    """
    global _current_chat_model, _current_embed_model

    # Import the modules that hold the model instances
    from src.api import routes_chat, routes_debug, routes_debug_citations

    changes: dict[str, dict[str, str]] = {}
    errors: list[str] = []

    if req.chat_model:
        old_model = _current_chat_model
        try:
            # Create new LLM instance
            # Note: ChatOllama validates the model on first use, not on creation
            new_llm = ChatOllama(
                model=req.chat_model,
                base_url=settings.ollama_base_url,
                temperature=0,
                num_predict=settings.ollama_num_predict,
            )

            # Test that the model works by doing a simple invocation
            # This will trigger model loading if not already loaded
            try:
                _ = new_llm.invoke([("human", "test")])
            except Exception as e:
                # If it's just a model loading issue, the model might still work
                # Check if it's a connection error vs model error
                error_str = str(e).lower()
                if "not found" in error_str or "does not exist" in error_str:
                    raise ValueError(f"Model '{req.chat_model}' not found in Ollama")
                # For other errors, the model might still be loading, continue anyway

            # Update all modules that use the LLM
            routes_chat._llm = new_llm
            routes_debug_citations._llm = new_llm

            _current_chat_model = req.chat_model
            changes["chat_model"] = {"from": old_model, "to": req.chat_model}

        except ValueError as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(f"Failed to set chat model '{req.chat_model}': {type(e).__name__}: {e}")

    if req.embed_model:
        old_model = _current_embed_model
        try:
            # Create new embedder
            new_embedder = make_embedder(settings.ollama_base_url, req.embed_model)

            # Test it works and get the embedding dimension
            try:
                test_vec = new_embedder.embed_query("test")
                new_dim = len(test_vec)
            except Exception as e:
                error_str = str(e).lower()
                if "not found" in error_str or "does not exist" in error_str:
                    raise ValueError(f"Model '{req.embed_model}' not found in Ollama")
                raise

            # Update ALL modules that use the embedder (including ingest!)
            from src.api import routes_ingest
            routes_chat._embedder = new_embedder
            routes_debug_citations._embedder = new_embedder
            routes_debug._embedder = new_embedder
            routes_ingest._embedder = new_embedder  # Critical: update ingest embedder too!

            # Check if FAISS index has incompatible dimension
            # Read dimension from disk if not loaded in memory
            old_faiss_dim = routes_ingest._faiss.dim
            if old_faiss_dim is None:
                # Try to read dimension from existing index on disk
                from pathlib import Path
                index_path = Path(settings.faiss_dir) / "index.faiss"
                if index_path.exists():
                    try:
                        import faiss
                        idx = faiss.read_index(str(index_path))
                        old_faiss_dim = idx.d
                    except Exception:
                        pass  # Index doesn't exist or is corrupted
            dimension_warning = None
            if old_faiss_dim is not None and old_faiss_dim != new_dim:
                dimension_warning = (
                    f"FAISS index has dimension {old_faiss_dim}, but new model produces {new_dim}. "
                    f"You must run /reset and /restart before ingesting new documents, "
                    f"or revert to an embedding model with {old_faiss_dim} dimensions."
                )

            _current_embed_model = req.embed_model
            changes["embed_model"] = {
                "from": old_model,
                "to": req.embed_model,
                "new_dimension": new_dim,
                "faiss_dimension": old_faiss_dim,
                "dimension_warning": dimension_warning,
            }

        except ValueError as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(f"Failed to set embed model '{req.embed_model}': {type(e).__name__}: {e}")

    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors, "changes": changes})

    # Persist the model changes to disk so they survive server restarts
    if changes:
        _save_persistent_config(
            chat_model=_current_chat_model if "chat_model" in changes else None,
            embed_model=_current_embed_model if "embed_model" in changes else None,
        )

    return {
        "success": True,
        "changes": changes,
        "current": get_current_models(),
    }
