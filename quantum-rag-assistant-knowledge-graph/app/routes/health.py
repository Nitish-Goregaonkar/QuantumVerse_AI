"""
Health check endpoints for monitoring and container orchestration.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from src.utils.config import settings

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    embedding_model: str


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version="2.5.0",
        model=settings.llm_model,
        embedding_model=settings.embedding_model,
    )