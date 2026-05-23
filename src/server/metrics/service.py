"""Service layer for site-wide metrics."""
import json

import asyncpg

from src.article_persistence.repositories.metrics_repository import MetricsRepository
from src.cache.cache_interface import CacheBackend
from src.server.metrics.schemas import MetricsResponse


CACHE_KEY = "metrics:counts"


class MetricsService:
    """Returns site-wide article and entity counts, cached for the hero section."""

    def __init__(
        self,
        repo: MetricsRepository | None = None,
        cache: CacheBackend | None = None,
    ):
        self._repo = repo or MetricsRepository()
        self._cache = cache

    async def get_metrics(self, conn: asyncpg.Connection) -> MetricsResponse:
        if self._cache is not None:
            cached = await self._cache.get(CACHE_KEY)
            if cached is not None:
                return MetricsResponse.model_validate(json.loads(cached))

        article_count = await self._repo.count_articles(conn)
        entity_count = await self._repo.count_entities_with_articles(conn)
        response = MetricsResponse(article_count=article_count, entity_count=entity_count)

        if self._cache is not None:
            await self._cache.set(
                CACHE_KEY,
                json.dumps(response.model_dump(mode="json")),
                ttl_seconds=300,
            )

        return response
