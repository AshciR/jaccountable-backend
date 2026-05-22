"""Service layer for article search."""
import json
from uuid import UUID

import asyncpg

from src.article_persistence.models.domain import ArticleSearchResult
from src.article_persistence.repositories.article_repository import ArticleRepository
from src.cache.cache_interface import CacheBackend
from src.server.articles.schemas import ArticleSearchParams


class ArticleSearchService:
    """Service that delegates article search to the repository, with optional caching.

    Caching strategy:
    - All requests (browse and q search) are cached.
    - Homepage entity-driven searches (e.g. q=INDECOM) are highly repeatable and
      benefit most from caching — they fire on every page load for each top entity.
    - Cache key includes all params (including q) to correctly differentiate results.
    """

    def __init__(
        self,
        repo: ArticleRepository | None = None,
        cache: CacheBackend | None = None,
    ):
        self._repo = repo or ArticleRepository()
        self._cache = cache

    async def search(
        self,
        conn: asyncpg.Connection,
        params: ArticleSearchParams,
    ) -> tuple[list[ArticleSearchResult], int]:
        """Search articles and return results with total count for pagination."""

        if self._cache is None:
            return await self._repo.search_articles(
                conn,
                q=params.q,
                from_date=params.from_date,
                to_date=params.to_date,
                include_full_text=params.include_full_text,
                page=params.page,
                page_size=params.page_size,
                sort=params.sort,
                order=params.order,
            )

        # Check if cache results are available
        cached: tuple[list[ArticleSearchResult], int] | None = await self._get_cached(params)
        if cached is not None:
            return cached

        # No cache, have to make DB call
        results, total = await self._repo.search_articles(
            conn,
            q=params.q,
            from_date=params.from_date,
            to_date=params.to_date,
            include_full_text=params.include_full_text,
            page=params.page,
            page_size=params.page_size,
            sort=params.sort,
            order=params.order,
        )
        await self._set_cached(params, results, total)

        return results, total

    async def get_by_public_id(
        self,
        conn: asyncpg.Connection,
        public_id: UUID,
    ) -> ArticleSearchResult | None:
        """Fetch a single article by public UUID, with entities and classifications."""
        if self._cache is None:
            return await self._repo.get_article_detail_by_public_id(conn, public_id)

        cache_key = f"article:detail:{public_id}"
        raw = await self._cache.get(cache_key)
        if raw:
            return ArticleSearchResult.model_validate(json.loads(raw))

        result = await self._repo.get_article_detail_by_public_id(conn, public_id)
        if result is not None:
            await self._cache.set(
                cache_key,
                json.dumps(result.model_dump(mode="json")),
                ttl_seconds=300,
            )
        return result

    async def get_related_articles(
        self,
        conn: asyncpg.Connection,
        public_id: UUID,
        limit: int = 5,
    ) -> list[ArticleSearchResult]:
        """Return articles related to the given article by shared entities."""
        if self._cache is None:
            return await self._repo.get_related_articles_by_public_id(conn, public_id, limit=limit)

        cache_key = f"article:related:{public_id}:limit={limit}"
        raw = await self._cache.get(cache_key)
        if raw:
            return [ArticleSearchResult.model_validate(r) for r in json.loads(raw)]

        results = await self._repo.get_related_articles_by_public_id(conn, public_id, limit=limit)
        await self._cache.set(
            cache_key,
            json.dumps([r.model_dump(mode="json") for r in results]),
            ttl_seconds=300,
        )
        return results

    async def _get_cached(
        self,
        params: ArticleSearchParams,
    ) -> tuple[list[ArticleSearchResult], int] | None:
        """Return cached results, or None on cache miss."""
        raw = await self._cache.get(self._build_cache_key(params))  # type: ignore[union-attr]
        if not raw:
            return None
        data = json.loads(raw)
        return (
            [ArticleSearchResult.model_validate(r) for r in data["results"]],
            data["total"],
        )

    async def _set_cached(
        self,
        params: ArticleSearchParams,
        results: list[ArticleSearchResult],
        total: int,
        ttl_seconds: int = 300,
    ) -> None:
        """Serialize and store search results in the cache."""
        payload = json.dumps({
            "results": [r.model_dump(mode="json") for r in results],
            "total": total,
        })
        await self._cache.set(self._build_cache_key(params), payload, ttl_seconds=ttl_seconds)  # type: ignore[union-attr]

    @staticmethod
    def _build_cache_key(params: ArticleSearchParams) -> str:
        """Build a deterministic cache key from all search request parameters."""
        return (
            f"articles:"
            f"q={params.q}:"
            f"p={params.page}:"
            f"ps={params.page_size}:"
            f"s={params.sort}:"
            f"o={params.order}:"
            f"fd={params.from_date}:"
            f"td={params.to_date}:"
            f"ft={params.include_full_text}"
        )
