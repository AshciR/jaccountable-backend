"""HTTP router for the metrics domain."""
from typing import Annotated

import asyncpg
from fastapi import APIRouter, Depends

from src.cache.cache_interface import CacheBackend
from src.server.dependencies import get_cache, get_db
from src.server.metrics.schemas import MetricsResponse
from src.server.metrics.service import MetricsService

router = APIRouter(prefix="/metrics", tags=["metrics"])


def _get_service(cache: Annotated[CacheBackend, Depends(get_cache)]) -> MetricsService:
    return MetricsService(cache=cache)


@router.get("", response_model=MetricsResponse)
async def get_metrics(
    conn: Annotated[asyncpg.Connection, Depends(get_db)],
    service: Annotated[MetricsService, Depends(_get_service)],
) -> MetricsResponse:
    return await service.get_metrics(conn)
