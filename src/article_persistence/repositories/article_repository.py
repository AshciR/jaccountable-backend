"""Repository for article database operations."""
import aiosql
import json
from datetime import datetime
from pathlib import Path
from uuid import UUID

import asyncpg

from src.article_persistence.models.domain import Article, ArticleSearchResult, SearchClassification


class ArticleRepository:
    """Repository for article database operations using aiosql."""

    def __init__(self):
        """Initialize the repository and load SQL queries."""
        # Load queries from the queries directory
        queries_path = Path(__file__).parent.parent / "queries"
        self.queries = aiosql.from_path(str(queries_path), "asyncpg")

    async def insert_article(
        self,
        conn: asyncpg.Connection,
        article: Article,
    ) -> Article:
        """
        Insert a new article into the database.

        Args:
            conn: Database connection to use for the query
            article: Article model with validated data

        Returns:
            Article: The inserted article with database-generated id

        Raises:
            asyncpg.UniqueViolationError: If article URL already exists
            ValueError: If article data fails validation
        """
        # Article model handles validation and provides fetched_at default
        result = await self.queries.insert_article(
            conn,
            url=article.url,
            title=article.title,
            section=article.section,
            published_date=article.published_date,
            fetched_at=article.fetched_at,
            full_text=article.full_text,
            news_source_id=article.news_source_id,
        )

        # Convert asyncpg.Record to Article model
        # Note: SQL query doesn't return full_text for performance,
        # so we use the original article's full_text value
        return Article(
            id=result['id'],
            public_id=result['public_id'],
            url=result['url'],
            title=result['title'],
            section=result['section'],
            published_date=result['published_date'],
            fetched_at=result['fetched_at'],
            full_text=article.full_text,
            news_source_id=result['news_source_id'],
        )

    async def get_existing_urls(
        self,
        conn: asyncpg.Connection,
        urls: list[str],
    ) -> set[str]:
        """
        Check which URLs from a list already exist in the database.

        Uses a single batch query for performance (60x-600x faster than
        individual queries for large batches).

        Args:
            conn: Database connection to use for the query
            urls: List of URLs to check

        Returns:
            Set of URLs that already exist in the database

        Example:
            >>> repo = ArticleRepository()
            >>> async with db_config.connection() as conn:
            ...     existing = await repo.get_existing_urls(
            ...         conn,
            ...         ["https://example.com/1", "https://example.com/2"]
            ...     )
            ...     print(existing)  # {"https://example.com/1"}
        """
        if not urls:
            return set()

        # Query using aiosql with PostgreSQL array syntax
        rows = await self.queries.get_existing_urls(conn, urls=urls)

        # Extract URLs from rows and return as set
        return {row["url"] for row in rows}

    async def get_by_public_id(
        self,
        conn: asyncpg.Connection,
        public_id: UUID,
    ) -> Article | None:
        """
        Retrieve an article by its public UUID.

        Args:
            conn: Database connection to use for the query
            public_id: The public UUID of the article

        Returns:
            Article if found, None otherwise
        """
        result = await self.queries.get_article_by_public_id(
            conn, public_id=public_id
        )
        if result is None:
            return None

        return Article(
            id=result['id'],
            public_id=result['public_id'],
            url=result['url'],
            title=result['title'],
            section=result['section'],
            published_date=result['published_date'],
            fetched_at=result['fetched_at'],
            full_text=result['full_text'],
            news_source_id=result['news_source_id'],
        )

    async def get_article_detail_by_public_id(
        self,
        conn: asyncpg.Connection,
        public_id: UUID,
    ) -> ArticleSearchResult | None:
        """Retrieve a single article with aggregated entities and classifications.

        Args:
            conn: Database connection to use for the query
            public_id: The public UUID of the article

        Returns:
            ArticleSearchResult if found (full_text populated, snippet=None), else None
        """
        row = await self.queries.get_article_detail_by_public_id(conn, public_id=public_id)
        if row is None:
            return None
        return _row_to_search_result(row)

    async def get_related_articles_by_public_id(
        self,
        conn: asyncpg.Connection,
        public_id: UUID,
        limit: int = 5,
    ) -> list[ArticleSearchResult]:
        """Retrieve articles related to the given article by shared entities.

        Results are ordered by number of shared entities descending, then by
        published_date descending. Articles from any news source may appear.

        Args:
            conn: Database connection to use for the query
            public_id: The public UUID of the source article
            limit: Maximum number of related articles to return

        Returns:
            List of ArticleSearchResult ordered by relatedness. Empty list if
            the article has no entities, no other articles share its entities,
            or the public_id does not exist.
        """
        rows = await self.queries.get_related_articles_by_public_id(
            conn, public_id=public_id, limit=limit
        )
        return [_row_to_search_result(row) for row in rows]

    async def search_articles(
        self,
        conn: asyncpg.Connection,
        q: str | None = None,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
        include_full_text: bool = False,
        page: int = 1,
        page_size: int = 20,
        sort: str = "relevance",
        order: str = "desc",
    ) -> tuple[list[ArticleSearchResult], int]:
        """Search articles using full-text search and/or entity name matching.

        The single `q` parameter searches both article text (PostgreSQL FTS via
        search_vector) and linked entity names (ILIKE). For example, searching
        "petrojam" returns articles where the title/body contains the term OR an
        associated entity is named "petrojam".

        When `q` is omitted the query returns all articles ordered by published_date
        (browse mode). Date filters apply in both modes.

        Args:
            conn: Database connection to use for the query.
            q: Search term. Matches article text (FTS) or entity names (ILIKE).
            from_date: Include only articles published on or after this date.
            to_date: Include only articles published on or before this date.
            include_full_text: When True, populate full_text on each result.
            page: 1-indexed page number for pagination.
            page_size: Number of results per page.
            sort: "relevance" (ts_rank, only meaningful with q) or "published_date".
            order: "asc" or "desc" sort direction (applies to published_date sort).

        Returns:
            Tuple of (results, total_count) where total_count is the count of all
            matching articles (ignoring pagination) for use in pagination metadata.
        """
        params: list = []
        param_idx = 0

        def track_param_count(value) -> str:
            nonlocal param_idx
            param_idx += 1
            params.append(value)
            return f"${param_idx}"

        use_fts = bool(q and q.strip())

        # --- SELECT clause ---
        full_text_col = "a.full_text" if include_full_text else "NULL::text as full_text"

        if use_fts:
            q_param = track_param_count(q)
            snippet_col = f"ts_headline('english', a.full_text, query, 'MaxWords=100, MinWords=50') as snippet"
        else:
            snippet_col = "LEFT(a.full_text, 600) as snippet"

        # --- Classifications CTE ---
        classifications_cte = """
            article_classifications AS (
                SELECT
                    article_id,
                    jsonb_agg(jsonb_build_object(
                        'classifier_type', classifier_type,
                        'confidence_score', confidence_score,
                        'reasoning', reasoning
                    )) AS classifications
                FROM classifications
                GROUP BY article_id
            )
        """

        select_clause = f"""
            a.public_id, a.url, a.title, a.section, a.published_date,
            ns.id as news_source_id,
            {snippet_col},
            {full_text_col},
            COALESCE(array_agg(DISTINCT e.name) FILTER (WHERE e.id IS NOT NULL), '{{}}') as entities,
            COALESCE(ac.classifications, '[]') as classifications
        """

        # --- FROM / JOIN clause ---
        if use_fts:
            from_clause = f"""
                FROM articles a
                CROSS JOIN plainto_tsquery('english', {q_param}) query
                JOIN news_sources ns ON a.news_source_id = ns.id
                LEFT JOIN article_classifications ac ON a.id = ac.article_id
                LEFT JOIN article_entities ae ON a.id = ae.article_id
                LEFT JOIN entities e ON ae.entity_id = e.id
            """
        else:
            from_clause = """
                FROM articles a
                JOIN news_sources ns ON a.news_source_id = ns.id
                LEFT JOIN article_classifications ac ON a.id = ac.article_id
                LEFT JOIN article_entities ae ON a.id = ae.article_id
                LEFT JOIN entities e ON ae.entity_id = e.id
            """

        # --- WHERE clause ---
        where_parts: list[str] = []

        if use_fts:
            where_parts.append(f"""(
                a.search_vector @@ query
                OR EXISTS (
                    SELECT 1 FROM article_entities ae2
                    JOIN entities e2 ON ae2.entity_id = e2.id
                    WHERE ae2.article_id = a.id
                    AND e2.name ILIKE '%' || {q_param} || '%'
                )
            )""")

        if from_date is not None:
            where_parts.append(f"a.published_date >= {track_param_count(from_date)}")
        if to_date is not None:
            where_parts.append(f"a.published_date <= {track_param_count(to_date)}")

        where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

        # --- GROUP BY ---
        group_by = "GROUP BY a.id, a.public_id, a.url, a.title, a.section, a.published_date, a.full_text, ns.id, ac.classifications"
        if use_fts:
            group_by += ", query"

        # --- ORDER BY ---
        if use_fts and sort == "relevance":
            order_by = "ORDER BY ts_rank(a.search_vector, query) DESC"
        else:
            direction = "ASC" if order == "asc" else "DESC"
            order_by = f"ORDER BY a.published_date {direction}"

        # --- LIMIT / OFFSET ---
        offset = (page - 1) * page_size
        limit_sql = f"LIMIT {track_param_count(page_size)} OFFSET {track_param_count(offset)}"

        # --- Main query ---
        main_sql = f"""
            WITH {classifications_cte}
            SELECT {select_clause}
            {from_clause}
            {where_sql}
            {group_by}
            {order_by}
            {limit_sql}
        """

        # --- Count query (same FROM/WHERE, no pagination) ---
        # params without the trailing limit/offset values
        count_params = params[:-2]
        count_sql = f"""
            WITH {classifications_cte}
            SELECT COUNT(DISTINCT a.id)
            {from_clause}
            {where_sql}
        """

        rows = await conn.fetch(main_sql, *params)
        total_count = await conn.fetchval(count_sql, *count_params)

        return [_row_to_search_result(row) for row in rows], (total_count or 0)

def _row_to_search_result(row: asyncpg.Record) -> ArticleSearchResult:
    """Convert a raw asyncpg Record from a search query into an ArticleSearchResult.

    Handles JSON parsing for the classifications column (returned as a JSON string
    by json_agg) and normalises the entities array_agg result to a plain Python list.
    """
    raw_cls = row["classifications"]
    cls_data = json.loads(raw_cls) if isinstance(raw_cls, str) else (raw_cls or [])
    return ArticleSearchResult(
        public_id=row["public_id"],
        url=row["url"],
        title=row["title"],
        section=row["section"],
        published_date=row["published_date"],
        news_source_id=row["news_source_id"],
        snippet=row["snippet"],
        entities=list(row["entities"] or []),
        classifications=[SearchClassification(**c) for c in cls_data],
        full_text=row["full_text"],
    )