from fastapi import APIRouter

from src.api.routes_chat import router as chat_router
from src.api.routes_chunks import router as chunks_router
from src.api.routes_debug import router as debug_router
from src.api.routes_debug_citations import router as debug_citations_router
from src.api.routes_ingest import router as ingest_router
from src.api.routes_models import router as models_router
from src.api.routes_stats import router as stats_router

router = APIRouter()
router.include_router(ingest_router)
router.include_router(chat_router)
router.include_router(debug_router)
router.include_router(chunks_router)
router.include_router(debug_citations_router)
router.include_router(stats_router)
router.include_router(models_router)

