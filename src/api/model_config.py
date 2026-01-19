"""Shared model configuration for persistent model settings.

This module provides functions to load the initial model configuration
from persistent storage, falling back to .env defaults if not available.

The persistent config file is stored in the data directory and survives
server restarts, allowing users to change models dynamically via the CLI
or API without editing .env files.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.core.config import settings

# Persistent configuration file path (stored alongside other data)
_MODEL_CONFIG_PATH = Path(settings.faiss_dir).parent.parent / "model_config.json"


def _load_persistent_config() -> dict[str, str]:
    """Load model configuration from persistent storage.

    Returns config from file if it exists, otherwise returns empty dict.
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


def get_initial_chat_model() -> str:
    """Get initial chat model, preferring persistent config over .env defaults."""
    persistent = _load_persistent_config()
    return persistent.get("chat_model") or settings.ollama_chat_model


def get_initial_embed_model() -> str:
    """Get initial embedding model, preferring persistent config over .env defaults."""
    persistent = _load_persistent_config()
    return persistent.get("embed_model") or settings.ollama_embed_model
