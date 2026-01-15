"""CLI Configuration management."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CLIConfig:
    """Configuration for the Hybrid RAG CLI."""
    
    # Server settings
    host: str = "127.0.0.1"
    port: int = 8000
    base_url: str = field(init=False)
    
    # Project paths
    project_root: Path = field(default_factory=lambda: Path.cwd())
    
    # Default query parameters
    session_id: str = "cli-session"
    bm25_k: int = 20
    vec_k: int = 20
    top_k: int = 8
    memory_k: int = 6
    
    # UI settings
    theme: str = "monokai"  # Color theme
    show_debug: bool = False
    
    def __post_init__(self):
        self.base_url = f"http://{self.host}:{self.port}"
    
    @classmethod
    def from_env(cls) -> "CLIConfig":
        """Create config from environment variables."""
        return cls(
            host=os.getenv("HYBRID_RAG_HOST", "127.0.0.1"),
            port=int(os.getenv("HYBRID_RAG_PORT", "8000")),
            session_id=os.getenv("HYBRID_RAG_SESSION", "cli-session"),
            project_root=Path(os.getenv("HYBRID_RAG_ROOT", Path.cwd())),
        )


# Global config instance
_config: Optional[CLIConfig] = None


def get_config() -> CLIConfig:
    """Get or create the global CLI configuration."""
    global _config
    if _config is None:
        _config = CLIConfig.from_env()
    return _config


def set_config(config: CLIConfig) -> None:
    """Set the global CLI configuration."""
    global _config
    _config = config
