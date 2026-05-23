"""Repository for site-wide count metrics."""
import asyncpg


class MetricsRepository:
    """Read-only repository returning aggregate counts for the site metrics endpoint."""

    async def count_articles(self, conn: asyncpg.Connection) -> int:
        return await conn.fetchval("SELECT COUNT(*) FROM articles") or 0

    async def count_entities_with_articles(self, conn: asyncpg.Connection) -> int:
        return await conn.fetchval(
            "SELECT COUNT(DISTINCT entity_id) FROM article_entities"
        ) or 0
