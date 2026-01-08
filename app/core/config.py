from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "gemma3:1b"
    ollama_embed_model: str = "mxbai-embed-large"
    ollama_num_predict: int = 512

    sqlite_path: str = "./data/db/app.db"
    schema_path: str = "./app/db/schema.sql"

    raw_dir: str = "./data/raw"
    faiss_dir: str = "./data/index/faiss"

    chunk_size: int = 1000
    chunk_overlap: int = 150

    use_faiss_gpu: bool = True
    faiss_gpu_device: int = 0


settings = Settings()
