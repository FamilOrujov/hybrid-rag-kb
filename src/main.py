from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.api.routes import router
from src.core.config import settings
from src.db.sqlite import init_db



@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db(settings.sqlite_path, settings.schema_path)
    yield


app = FastAPI(title="Hybrid RAG KB (BM25 + FAISS GPU)", lifespan=lifespan)
app.include_router(router)



@app.get("/health")
def health():
    return {"status": "ok"}

