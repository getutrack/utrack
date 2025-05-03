from fastapi import APIRouter

from risk_analyzer.api.endpoints import (
    hybrid_search,
    risk,
    analysis,
    embedding
)

api_router = APIRouter()

# Hybrid search endpoints
api_router.include_router(hybrid_search.router, prefix="/search", tags=["search"])

# Risk analysis endpoints
api_router.include_router(risk.router, prefix="/risk", tags=["risk"])

# Team and workflow analysis endpoints
api_router.include_router(analysis.router, prefix="/analysis", tags=["analysis"])

# Embedding endpoints
api_router.include_router(embedding.router, prefix="/embedding", tags=["embedding"]) 