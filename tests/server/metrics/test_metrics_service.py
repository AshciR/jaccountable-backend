"""Tests for MetricsService.get_metrics()."""
import asyncpg

from src.cache.in_memory import InMemoryCache
from src.server.metrics.service import CACHE_KEY, MetricsService
from tests.article_persistence.utils import (
    create_test_article_entity,
    create_test_entity,
    insert_article_with_date,
)


class TestMetricsServiceHappyPath:
    async def test_empty_db_returns_zero_counts(self, db_connection: asyncpg.Connection):
        # Given: an empty database
        # When: fetching metrics
        service = MetricsService()
        result = await service.get_metrics(db_connection)

        # Then: both counts are zero
        assert result.article_count == 0
        assert result.entity_count == 0

    async def test_counts_articles_and_linked_entities(self, db_connection: asyncpg.Connection):
        # Given: 3 articles, 2 entities each linked to one article
        articles = [
            await insert_article_with_date(db_connection, url=f"https://example.com/m-{i}")
            for i in range(3)
        ]
        entity_a = await create_test_entity(
            db_connection, name="Entity A", normalized_name="entity a metrics"
        )
        entity_b = await create_test_entity(
            db_connection, name="Entity B", normalized_name="entity b metrics"
        )
        await create_test_article_entity(db_connection, articles[0].id, entity_a.id)
        await create_test_article_entity(db_connection, articles[1].id, entity_b.id)

        # When: fetching metrics
        service = MetricsService()
        result = await service.get_metrics(db_connection)

        # Then: counts reflect the inserted data
        assert result.article_count == 3
        assert result.entity_count == 2

    async def test_orphan_entity_is_excluded_from_count(self, db_connection: asyncpg.Connection):
        # Given: one linked entity and one orphaned entity (no article_entities row)
        article = await insert_article_with_date(db_connection, url="https://example.com/m-orph-1")
        linked = await create_test_entity(
            db_connection, name="Linked", normalized_name="linked metrics orphan"
        )
        await create_test_entity(
            db_connection, name="Orphan", normalized_name="orphan metrics orphan"
        )
        await create_test_article_entity(db_connection, article.id, linked.id)

        # When: fetching metrics
        service = MetricsService()
        result = await service.get_metrics(db_connection)

        # Then: only the linked entity is counted
        assert result.article_count == 1
        assert result.entity_count == 1

    async def test_same_entity_linked_to_multiple_articles_counts_once(
        self, db_connection: asyncpg.Connection
    ):
        # Given: one entity linked to two articles (two article_entities rows)
        entity = await create_test_entity(
            db_connection, name="Recurring", normalized_name="recurring metrics multi"
        )
        for i in range(2):
            article = await insert_article_with_date(
                db_connection, url=f"https://example.com/m-multi-{i}"
            )
            await create_test_article_entity(db_connection, article.id, entity.id)

        # When: fetching metrics
        service = MetricsService()
        result = await service.get_metrics(db_connection)

        # Then: entity_count is 1 (DISTINCT entity_id)
        assert result.article_count == 2
        assert result.entity_count == 1


class TestMetricsServiceCaching:
    async def test_first_call_populates_cache(self, db_connection: asyncpg.Connection):
        # Given: an article + linked entity and a fresh in-memory cache
        article = await insert_article_with_date(db_connection, url="https://example.com/m-cache-1")
        entity = await create_test_entity(
            db_connection, name="Cache E", normalized_name="cache metrics first"
        )
        await create_test_article_entity(db_connection, article.id, entity.id)

        cache = InMemoryCache()
        service = MetricsService(cache=cache)

        # When: get_metrics is called once
        await service.get_metrics(db_connection)

        # Then: the metrics key is in the cache
        assert cache.size() == 1
        assert await cache.get(CACHE_KEY) is not None

    async def test_second_call_served_from_cache(self, db_connection: asyncpg.Connection):
        # Given: a primed cache populated by a first call
        article = await insert_article_with_date(db_connection, url="https://example.com/m-cache-2")
        entity = await create_test_entity(
            db_connection, name="Cache E2", normalized_name="cache metrics second"
        )
        await create_test_article_entity(db_connection, article.id, entity.id)
        cache = InMemoryCache()
        service = MetricsService(cache=cache)
        first_metrics = await service.get_metrics(db_connection)

        # When: the DB state changes but cache is still warm, and we call again
        new_article = await insert_article_with_date(
            db_connection, url="https://example.com/m-cache-2-extra"
        )
        await create_test_article_entity(db_connection, new_article.id, entity.id)
        second_metrics = await service.get_metrics(db_connection)

        # Then: the second call returns the cached (stale) result, proving cache was used
        assert second_metrics.article_count == first_metrics.article_count
        assert second_metrics.entity_count == first_metrics.entity_count

    async def test_service_without_cache_still_returns_results(
        self, db_connection: asyncpg.Connection
    ):
        # Given: a service with no cache backend
        article = await insert_article_with_date(db_connection, url="https://example.com/m-nocache")
        entity = await create_test_entity(
            db_connection, name="No Cache", normalized_name="no cache metrics"
        )
        await create_test_article_entity(db_connection, article.id, entity.id)
        service = MetricsService(cache=None)

        # When: get_metrics is called
        result = await service.get_metrics(db_connection)

        # Then: results are computed from the DB
        assert result.article_count >= 1
        assert result.entity_count >= 1
