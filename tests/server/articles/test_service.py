"""Tests for ArticleSearchService.search()."""

import asyncpg
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from src.article_persistence.models.domain import Article, Classification
from src.article_persistence.repositories.article_repository import ArticleRepository
from src.article_persistence.repositories.classification_repository import ClassificationRepository
from src.cache.in_memory import InMemoryCache
from src.server.articles.schemas import ArticleSearchParams
from src.server.articles.service import ArticleSearchService
from tests.article_persistence.utils import (
    create_test_article_entity,
    create_test_entity,
    delete_article,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _insert_article_with_text(
    conn: asyncpg.Connection,
    url: str,
    title: str,
    full_text: str,
    published_date: datetime | None = None,
    news_source_id: int = 1,
) -> Article:
    """Insert an article with full_text and optional published_date."""
    repo = ArticleRepository()
    article = Article(
        url=url,
        title=title,
        section="news",
        full_text=full_text,
        published_date=published_date,
        news_source_id=news_source_id,
    )
    return await repo.insert_article(conn, article)


async def _insert_classification(
    conn: asyncpg.Connection,
    article_id: int,
    classifier_type: str = "CORRUPTION",
    confidence_score: float = 0.9,
    reasoning: str | None = None,
) -> Classification:
    repo = ClassificationRepository()
    classification = Classification(
        article_id=article_id,
        classifier_type=classifier_type,
        confidence_score=confidence_score,
        reasoning=reasoning,
        model_name="gpt-4o-mini",
    )
    return await repo.insert_classification(conn, classification)


# ---------------------------------------------------------------------------
# Full-text search tests
# ---------------------------------------------------------------------------

class TestSearchArticlesFTS:
    """Full-text search via the q parameter."""

    async def test_search_keyword_in_title_returns_match(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article whose title contains "corruption"
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/fts-title",
            title="Corruption probe at ministry",
            full_text="Investigators found evidence of wrongdoing.",
        )
        service = ArticleSearchService()

        # When: we search for "corruption"
        results, total = await service.search(
            db_connection, ArticleSearchParams(q="corruption")
        )

        # Then: the article is returned
        assert total >= 1
        urls = [r.url for r in results]
        assert "https://example.com/fts-title" in urls

    async def test_search_keyword_in_full_text_returns_match(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article whose body (not title) contains "embezzlement"
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/fts-body",
            title="Minister faces inquiry",
            full_text="Evidence of embezzlement was uncovered during the audit.",
        )
        service = ArticleSearchService()

        # When: we search for "embezzlement"
        results, total = await service.search(
            db_connection, ArticleSearchParams(q="embezzlement")
        )

        # Then: the article is returned
        assert total >= 1
        urls = [r.url for r in results]
        assert "https://example.com/fts-body" in urls

    async def test_search_returns_snippet(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article that matches the search query
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/snippet-test",
            title="Bribery scandal exposed",
            full_text="Officials accepted bribery payments over several years.",
        )
        service = ArticleSearchService()

        # When: we search for "bribery"
        results, _ = await service.search(
            db_connection, ArticleSearchParams(q="bribery")
        )

        # Then: the matching result has a non-null snippet
        match = next(r for r in results if r.url == "https://example.com/snippet-test")
        assert match.snippet is not None
        assert len(match.snippet) > 0

    async def test_search_no_match_returns_empty(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: the database contains articles
        service = ArticleSearchService()

        # When: we search for an extremely unlikely term
        results, total = await service.search(
            db_connection, ArticleSearchParams(q="xyzzyunlikelytermzqq")
        )

        # Then: no results and count is zero
        assert results == []
        assert total == 0

    async def test_search_total_count_reflects_all_matches(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: three articles, two matching "accountability", one not
        for i in range(2):
            await _insert_article_with_text(
                db_connection,
                url=f"https://example.com/count-match-{i}",
                title=f"Government accountability report {i}",
                full_text="This concerns accountability in public office.",
            )
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/count-nomatch",
            title="Sports news today",
            full_text="The cricket team won the match.",
        )
        service = ArticleSearchService()

        # When: we search with page_size=1
        results, total = await service.search(
            db_connection, ArticleSearchParams(q="accountability", page=1, page_size=1)
        )

        # Then: only 1 result returned, but total >= 2
        assert len(results) == 1
        assert total >= 2


# ---------------------------------------------------------------------------
# Browse mode (no q)
# ---------------------------------------------------------------------------

class TestSearchArticlesBrowse:
    """Behaviour when q is omitted — returns all articles ordered by date."""

    async def test_browse_without_q_returns_articles(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article exists
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/browse-test",
            title="Browse mode article",
            full_text="Content for browse.",
        )
        service = ArticleSearchService()

        # When: we search with no q
        results, total = await service.search(db_connection, ArticleSearchParams())

        # Then: results are returned and snippet is the first 600 chars of full_text
        assert total >= 1
        match = next(
            (r for r in results if r.url == "https://example.com/browse-test"), None
        )
        assert match is not None
        assert match.snippet == "Content for browse."

    async def test_browse_default_sort_is_published_date_desc(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: two articles with different published dates
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/browse-old",
            title="Older article",
            full_text="Older content.",
            published_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
        )
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/browse-new",
            title="Newer article",
            full_text="Newer content.",
            published_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        service = ArticleSearchService()

        # When: we browse with default sort
        results, _ = await service.search(db_connection, ArticleSearchParams())

        # Then: newer article appears before older article
        urls = [r.url for r in results]
        newer_idx = next(
            (i for i, u in enumerate(urls) if u == "https://example.com/browse-new"), None
        )
        older_idx = next(
            (i for i, u in enumerate(urls) if u == "https://example.com/browse-old"), None
        )
        assert newer_idx is not None
        assert older_idx is not None
        assert newer_idx < older_idx

    async def test_browse_sort_asc_returns_oldest_first(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: two articles with known published dates
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/asc-old",
            title="Earliest article",
            full_text="Content.",
            published_date=datetime(2022, 3, 1, tzinfo=timezone.utc),
        )
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/asc-new",
            title="Latest article",
            full_text="Content.",
            published_date=datetime(2025, 3, 1, tzinfo=timezone.utc),
        )
        service = ArticleSearchService()

        # When: we browse with sort=published_date, order=asc
        results, _ = await service.search(
            db_connection,
            ArticleSearchParams(sort="published_date", order="asc"),
        )

        # Then: oldest article with a date comes before newer
        dated = [r for r in results if r.published_date is not None]
        dates = [r.published_date for r in dated]
        assert dates == sorted(dates)


# ---------------------------------------------------------------------------
# Date filter tests
# ---------------------------------------------------------------------------

class TestSearchArticlesDateFilter:
    """Date range filtering via from_date and to_date."""

    async def test_from_date_excludes_older_articles(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: articles before and after a cutoff date
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/date-before",
            title="Old article",
            full_text="Content.",
            published_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/date-after",
            title="Recent article",
            full_text="Content.",
            published_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        service = ArticleSearchService()

        # When: we filter from 2023-01-01
        results, _ = await service.search(
            db_connection,
            ArticleSearchParams(from_date=datetime(2023, 1, 1, tzinfo=timezone.utc)),
        )

        # Then: old article is excluded, recent article is included
        urls = [r.url for r in results]
        assert "https://example.com/date-before" not in urls
        assert "https://example.com/date-after" in urls

    async def test_to_date_excludes_newer_articles(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: articles with different dates
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/todate-old",
            title="Old article",
            full_text="Content.",
            published_date=datetime(2019, 6, 1, tzinfo=timezone.utc),
        )
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/todate-new",
            title="New article",
            full_text="Content.",
            published_date=datetime(2025, 6, 1, tzinfo=timezone.utc),
        )
        service = ArticleSearchService()

        # When: we filter to_date=2020-01-01
        results, _ = await service.search(
            db_connection,
            ArticleSearchParams(to_date=datetime(2020, 1, 1, tzinfo=timezone.utc)),
        )

        # Then: new article is excluded, old article is included
        urls = [r.url for r in results]
        assert "https://example.com/todate-new" not in urls
        assert "https://example.com/todate-old" in urls

    async def test_from_date_boundary_is_inclusive(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article published exactly on the from_date boundary
        boundary = datetime(2024, 3, 15, tzinfo=timezone.utc)
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/boundary-test",
            title="Boundary date article",
            full_text="Content.",
            published_date=boundary,
        )
        service = ArticleSearchService()

        # When: we query with from_date equal to the article's published_date
        results, _ = await service.search(
            db_connection, ArticleSearchParams(from_date=boundary)
        )

        # Then: the article is included (>= is inclusive)
        urls = [r.url for r in results]
        assert "https://example.com/boundary-test" in urls

    async def test_combined_date_range_filters_correctly(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: articles outside and inside a date window
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/range-too-old",
            title="Too old",
            full_text="Content.",
            published_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/range-in-window",
            title="In window",
            full_text="Content.",
            published_date=datetime(2022, 6, 1, tzinfo=timezone.utc),
        )
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/range-too-new",
            title="Too new",
            full_text="Content.",
            published_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        service = ArticleSearchService()

        # When: we filter with from_date=2022-01-01 and to_date=2023-01-01
        results, _ = await service.search(
            db_connection,
            ArticleSearchParams(
                from_date=datetime(2022, 1, 1, tzinfo=timezone.utc),
                to_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            ),
        )

        # Then: only the in-window article is present
        urls = [r.url for r in results]
        assert "https://example.com/range-in-window" in urls
        assert "https://example.com/range-too-old" not in urls
        assert "https://example.com/range-too-new" not in urls


# ---------------------------------------------------------------------------
# Entity search tests
# ---------------------------------------------------------------------------

class TestSearchArticlesEntitySearch:
    """q matches entity names as well as article text."""

    async def test_search_by_entity_name_returns_article(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article linked to an entity named "Petrojam"
        # but the article text does NOT contain the word "petrojam"
        article = await _insert_article_with_text(
            db_connection,
            url="https://example.com/entity-search",
            title="State enterprise under scrutiny",
            full_text="The auditor general reviewed the company finances.",
        )
        entity = await create_test_entity(
            db_connection,
            name="Petrojam",
            normalized_name="petrojam",
        )
        await create_test_article_entity(db_connection, article.id, entity.id)
        service = ArticleSearchService()

        # When: we search for "Petrojam"
        results, total = await service.search(
            db_connection, ArticleSearchParams(q="Petrojam")
        )

        # Then: the article is returned even though text doesn't contain the term
        assert total >= 1
        urls = [r.url for r in results]
        assert "https://example.com/entity-search" in urls

    async def test_entity_search_is_case_insensitive(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an entity named "Andrew Holness"
        article = await _insert_article_with_text(
            db_connection,
            url="https://example.com/entity-case",
            title="Cabinet reshuffle announced",
            full_text="A significant change in government positions.",
        )
        entity = await create_test_entity(
            db_connection,
            name="Andrew Holness",
            normalized_name="andrew holness",
        )
        await create_test_article_entity(db_connection, article.id, entity.id)
        service = ArticleSearchService()

        # When: we search with lowercase "holness"
        results, total = await service.search(
            db_connection, ArticleSearchParams(q="holness")
        )

        # Then: the article is still returned (ILIKE is case-insensitive)
        assert total >= 1
        urls = [r.url for r in results]
        assert "https://example.com/entity-case" in urls

    async def test_entity_search_returns_all_entities_for_article(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article linked to two entities, search matches only one entity name
        article = await _insert_article_with_text(
            db_connection,
            url="https://example.com/entity-full-list",
            title="Multi-entity article",
            full_text="An investigation involving multiple parties.",
        )
        entity1 = await create_test_entity(
            db_connection,
            name="Petrojam Limited",
            normalized_name="petrojam limited",
        )
        entity2 = await create_test_entity(
            db_connection,
            name="Ministry of Energy",
            normalized_name="ministry of energy",
        )
        await create_test_article_entity(db_connection, article.id, entity1.id)
        await create_test_article_entity(db_connection, article.id, entity2.id)
        service = ArticleSearchService()

        # When: we search for "petrojam" (matches entity1, not entity2)
        results, _ = await service.search(
            db_connection, ArticleSearchParams(q="petrojam")
        )

        # Then: the result for this article contains BOTH entity names
        match = next(
            (r for r in results if r.url == "https://example.com/entity-full-list"), None
        )
        assert match is not None
        assert "Petrojam Limited" in match.entities
        assert "Ministry of Energy" in match.entities


# ---------------------------------------------------------------------------
# Pagination tests
# ---------------------------------------------------------------------------

class TestSearchArticlesPagination:
    """Offset/limit pagination via page and page_size."""

    async def test_page_size_limits_results(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: five articles that all share a unique keyword
        for i in range(5):
            await _insert_article_with_text(
                db_connection,
                url=f"https://example.com/paginate-{i}",
                title=f"Uniquepaginateword article {i}",
                full_text="Content about uniquepaginateword.",
            )
        service = ArticleSearchService()

        # When: we search with page_size=3
        results, _ = await service.search(
            db_connection,
            ArticleSearchParams(q="uniquepaginateword", page=1, page_size=3),
        )

        # Then: exactly 3 results are returned
        assert len(results) == 3

    async def test_page_2_returns_next_items(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: four articles matching "offsetkeyword"
        for i in range(4):
            await _insert_article_with_text(
                db_connection,
                url=f"https://example.com/offset-{i}",
                title=f"Offsetkeyword article {i}",
                full_text="Content about offsetkeyword.",
            )
        service = ArticleSearchService()

        # When: page 1 and page 2 are fetched with page_size=2
        page1_results, _ = await service.search(
            db_connection,
            ArticleSearchParams(q="offsetkeyword", page=1, page_size=2),
        )
        page2_results, _ = await service.search(
            db_connection,
            ArticleSearchParams(q="offsetkeyword", page=2, page_size=2),
        )

        # Then: pages are non-overlapping
        page1_urls = {r.url for r in page1_results}
        page2_urls = {r.url for r in page2_results}
        assert page1_urls.isdisjoint(page2_urls)

    async def test_total_count_reflects_all_matches_not_page(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: six articles matching "totalcountword"
        for i in range(6):
            await _insert_article_with_text(
                db_connection,
                url=f"https://example.com/totalcount-{i}",
                title=f"Totalcountword article {i}",
                full_text="Content about totalcountword.",
            )
        service = ArticleSearchService()

        # When: we fetch page 1 with page_size=2
        results, total = await service.search(
            db_connection,
            ArticleSearchParams(q="totalcountword", page=1, page_size=2),
        )

        # Then: 2 results returned but total reflects all 6 matches
        assert len(results) == 2
        assert total == 6


# ---------------------------------------------------------------------------
# Aggregate tests (classifications + entities)
# ---------------------------------------------------------------------------

class TestSearchArticlesAggregates:
    """classifications and entities are properly aggregated per result."""

    async def test_classifications_are_aggregated(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article with two classifications
        article = await _insert_article_with_text(
            db_connection,
            url="https://example.com/agg-classifications",
            title="Classified article aggregation",
            full_text="This article was classified by two classifiers.",
        )
        await _insert_classification(
            db_connection, article.id, classifier_type="CORRUPTION", confidence_score=0.9
        )
        await _insert_classification(
            db_connection, article.id, classifier_type="HURRICANE_RELIEF", confidence_score=0.7
        )
        service = ArticleSearchService()

        # When: we search for this article
        results, _ = await service.search(
            db_connection, ArticleSearchParams(q="aggregation")
        )

        # Then: both classifications appear in the result
        match = next(
            (r for r in results if r.url == "https://example.com/agg-classifications"), None
        )
        assert match is not None
        assert len(match.classifications) == 2
        types = {c.classifier_type for c in match.classifications}
        assert "CORRUPTION" in types
        assert "HURRICANE_RELIEF" in types

    async def test_classification_reasoning_is_included(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article with a classification that has reasoning
        article = await _insert_article_with_text(
            db_connection,
            url="https://example.com/reasoning-test",
            title="Article with reasoning",
            full_text="This article has a classification with reasoning attached.",
        )
        await _insert_classification(
            db_connection,
            article.id,
            classifier_type="CORRUPTION",
            confidence_score=0.85,
            reasoning="Article directly discusses government corruption.",
        )
        service = ArticleSearchService()

        # When: we search for this article
        results, _ = await service.search(
            db_connection, ArticleSearchParams(q="reasoning")
        )

        # Then: the classification includes the reasoning text
        match = next(
            (r for r in results if r.url == "https://example.com/reasoning-test"), None
        )
        assert match is not None
        assert len(match.classifications) == 1
        cls = match.classifications[0]
        assert cls.reasoning == "Article directly discusses government corruption."

    async def test_entities_are_aggregated(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article linked to two entities
        article = await _insert_article_with_text(
            db_connection,
            url="https://example.com/agg-entities",
            title="Entities aggregation test",
            full_text="Multiple entities are associated with this story.",
        )
        e1 = await create_test_entity(
            db_connection, name="Alpha Corp", normalized_name="alpha corp"
        )
        e2 = await create_test_entity(
            db_connection, name="Beta Ltd", normalized_name="beta ltd"
        )
        await create_test_article_entity(db_connection, article.id, e1.id)
        await create_test_article_entity(db_connection, article.id, e2.id)
        service = ArticleSearchService()

        # When: we search for the article
        results, _ = await service.search(
            db_connection, ArticleSearchParams(q="aggregation")
        )

        # Then: both entity names appear in the result
        match = next(
            (r for r in results if r.url == "https://example.com/agg-entities"), None
        )
        assert match is not None
        assert "Alpha Corp" in match.entities
        assert "Beta Ltd" in match.entities

    async def test_article_without_classifications_has_empty_list(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article with no classifications
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/no-classifications",
            title="Unclassified article noclassword",
            full_text="This article has no classifications.",
        )
        service = ArticleSearchService()

        # When: we search for it
        results, _ = await service.search(
            db_connection, ArticleSearchParams(q="noclassword")
        )

        # Then: classifications is an empty list (not None)
        match = next(
            (r for r in results if r.url == "https://example.com/no-classifications"), None
        )
        assert match is not None
        assert match.classifications == []

    async def test_article_without_entities_has_empty_list(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article with no entities
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/no-entities",
            title="No entities article noentityword",
            full_text="This article has no linked entities.",
        )
        service = ArticleSearchService()

        # When: we search for it
        results, _ = await service.search(
            db_connection, ArticleSearchParams(q="noentityword")
        )

        # Then: entities is an empty list (not None)
        match = next(
            (r for r in results if r.url == "https://example.com/no-entities"), None
        )
        assert match is not None
        assert match.entities == []


# ---------------------------------------------------------------------------
# include_full_text toggle
# ---------------------------------------------------------------------------

class TestSearchArticlesFullText:
    """full_text field is controlled by include_full_text parameter."""

    async def test_include_full_text_false_returns_none(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article with full_text content
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/ft-false",
            title="Full text false test ftfalseword",
            full_text="This is the full article text that should be hidden.",
        )
        service = ArticleSearchService()

        # When: we search without include_full_text (default False)
        results, _ = await service.search(
            db_connection,
            ArticleSearchParams(q="ftfalseword", include_full_text=False),
        )

        # Then: full_text is None on the result
        match = next(
            (r for r in results if r.url == "https://example.com/ft-false"), None
        )
        assert match is not None
        assert match.full_text is None

    async def test_include_full_text_true_returns_content(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article with full_text content
        expected_text = "This is the full article text that should be visible."
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/ft-true",
            title="Full text true test fttrueword",
            full_text=expected_text,
        )
        service = ArticleSearchService()

        # When: we search with include_full_text=True
        results, _ = await service.search(
            db_connection,
            ArticleSearchParams(q="fttrueword", include_full_text=True),
        )

        # Then: full_text is populated on the result
        match = next(
            (r for r in results if r.url == "https://example.com/ft-true"), None
        )
        assert match is not None
        assert match.full_text == expected_text


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestSearchArticlesEdgeCases:
    """Edge cases and combined filter scenarios."""

    async def test_empty_string_q_treated_as_browse(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article exists
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/empty-q",
            title="Empty q test article",
            full_text="Content.",
        )
        service = ArticleSearchService()

        # When: we search with q="" (empty string)
        results, total = await service.search(
            db_connection, ArticleSearchParams(q="")
        )

        # Then: falls back to browse mode (does not crash), snippet is the first 600 chars of full_text
        assert total >= 1
        assert all(r.snippet is not None for r in results)

    async def test_all_filters_combined(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: a matching article (text + date window)
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/combined",
            title="Combined filters uniquecombinedword",
            full_text="Content with uniquecombinedword in range.",
            published_date=datetime(2023, 6, 1, tzinfo=timezone.utc),
        )
        # And an article outside the date window with the same keyword
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/combined-out",
            title="Combined filters uniquecombinedword outside range",
            full_text="Content with uniquecombinedword outside range.",
            published_date=datetime(2021, 1, 1, tzinfo=timezone.utc),
        )
        service = ArticleSearchService()

        # When: we search with q + date range
        results, total = await service.search(
            db_connection,
            ArticleSearchParams(
                q="uniquecombinedword",
                from_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
                to_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
        )

        # Then: only the in-range article is returned
        urls = [r.url for r in results]
        assert "https://example.com/combined" in urls
        assert "https://example.com/combined-out" not in urls
        assert total == 1

    async def test_relevance_sort_without_q_falls_back_to_date(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: two articles with different published dates
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/relevance-fallback-old",
            title="Relevance fallback old",
            full_text="Content.",
            published_date=datetime(2021, 1, 1, tzinfo=timezone.utc),
        )
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/relevance-fallback-new",
            title="Relevance fallback new",
            full_text="Content.",
            published_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        service = ArticleSearchService()

        # When: no q but sort=relevance (should fall back to published_date desc)
        results, _ = await service.search(
            db_connection,
            ArticleSearchParams(q=None, sort="relevance", order="desc"),
        )

        # Then: newer article comes before older (date desc fallback)
        urls = [r.url for r in results]
        new_idx = next(
            (i for i, u in enumerate(urls) if u == "https://example.com/relevance-fallback-new"),
            None,
        )
        old_idx = next(
            (i for i, u in enumerate(urls) if u == "https://example.com/relevance-fallback-old"),
            None,
        )
        assert new_idx is not None
        assert old_idx is not None
        assert new_idx < old_idx


# ---------------------------------------------------------------------------
# Cache behaviour tests
# ---------------------------------------------------------------------------


class TestArticleSearchServiceCaching:
    """Cache integration: miss, hit, and bypass for search queries."""

    async def test_browse_populates_cache_on_first_call(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article and a fresh cache
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/cache-browse",
            title="Cache test article",
            full_text="Content for cache population test.",
        )
        cache = InMemoryCache()
        service = ArticleSearchService(cache=cache)

        # When: we call search in browse mode (no q)
        await service.search(db_connection, ArticleSearchParams())

        # Then: the cache has exactly one entry
        assert cache.size() == 1

    async def test_different_browse_params_produce_separate_cache_entries(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article and a service with a cache
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/cache-browse-params",
            title="Cache params article",
            full_text="Content for cache key differentiation test.",
        )
        cache = InMemoryCache()
        service = ArticleSearchService(cache=cache)

        # When: we browse with two different page sizes
        await service.search(db_connection, ArticleSearchParams(page_size=10))
        await service.search(db_connection, ArticleSearchParams(page_size=20))

        # Then: each distinct param combination produces its own cache entry
        assert cache.size() == 2

    async def test_search_with_q_populates_cache(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article and a fresh cache
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/cache-search-q",
            title="Cache population for corruption search",
            full_text="This article discusses corruption in public office.",
        )
        cache = InMemoryCache()
        service = ArticleSearchService(cache=cache)

        # When: we call search with a free-text query (entity-driven homepage search)
        await service.search(db_connection, ArticleSearchParams(q="corruption"))

        # Then: the result is cached (entity searches are highly repeatable)
        assert cache.size() == 1

    async def test_service_without_cache_still_returns_results(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article and a service with no cache (cache=None)
        await _insert_article_with_text(
            db_connection,
            url="https://example.com/cache-none",
            title="No cache service test",
            full_text="Content for no-cache service test.",
        )
        service = ArticleSearchService(cache=None)

        # When: we browse
        results, total = await service.search(db_connection, ArticleSearchParams())

        # Then: results are returned normally
        assert total >= 1
        assert any(r.url == "https://example.com/cache-none" for r in results)


# ---------------------------------------------------------------------------
# Article detail tests
# ---------------------------------------------------------------------------


class TestArticleDetailService:
    """ArticleSearchService.get_by_public_id() — fetch a single article by UUID."""

    async def test_get_by_public_id_returns_article(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article with known content
        expected_text = "Full content for detail test."
        article = await _insert_article_with_text(
            db_connection,
            url="https://example.com/detail-basic",
            title="Detail basic article",
            full_text=expected_text,
        )
        service = ArticleSearchService()

        # When: we fetch by public_id
        result = await service.get_by_public_id(db_connection, article.public_id)

        # Then: the correct article is returned with full_text populated
        assert result is not None
        assert result.url == "https://example.com/detail-basic"
        assert result.title == "Detail basic article"
        assert result.full_text == expected_text

    async def test_get_by_public_id_not_found_returns_none(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: a UUID that does not exist in the database
        from uuid import uuid4
        non_existent = uuid4()
        service = ArticleSearchService()

        # When: we fetch by that UUID
        result = await service.get_by_public_id(db_connection, non_existent)

        # Then: None is returned
        assert result is None

    async def test_get_by_public_id_snippet_is_none(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article
        article = await _insert_article_with_text(
            db_connection,
            url="https://example.com/detail-snippet",
            title="Snippet detail article",
            full_text="Content.",
        )
        service = ArticleSearchService()

        # When: we fetch by public_id
        result = await service.get_by_public_id(db_connection, article.public_id)

        # Then: snippet is None (it is a search artifact, not relevant to detail view)
        assert result is not None
        assert result.snippet is None

    async def test_get_by_public_id_entities_aggregated(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article linked to two entities
        article = await _insert_article_with_text(
            db_connection,
            url="https://example.com/detail-entities",
            title="Detail entities article",
            full_text="Content.",
        )
        e1 = await create_test_entity(db_connection, name="Petrojam", normalized_name="petrojam")
        e2 = await create_test_entity(db_connection, name="INDECOM", normalized_name="indecom")
        await create_test_article_entity(db_connection, article.id, e1.id)
        await create_test_article_entity(db_connection, article.id, e2.id)
        service = ArticleSearchService()

        # When: we fetch by public_id
        result = await service.get_by_public_id(db_connection, article.public_id)

        # Then: both entities appear in the result
        assert result is not None
        assert "Petrojam" in result.entities
        assert "INDECOM" in result.entities

    async def test_get_by_public_id_classifications_aggregated(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article with two classifications
        article = await _insert_article_with_text(
            db_connection,
            url="https://example.com/detail-classifications",
            title="Detail classifications article",
            full_text="Content.",
        )
        await _insert_classification(db_connection, article.id, classifier_type="CORRUPTION", confidence_score=0.9)
        await _insert_classification(db_connection, article.id, classifier_type="HURRICANE_RELIEF", confidence_score=0.7)
        service = ArticleSearchService()

        # When: we fetch by public_id
        result = await service.get_by_public_id(db_connection, article.public_id)

        # Then: both classifications are returned
        assert result is not None
        assert len(result.classifications) == 2
        types = {c.classifier_type for c in result.classifications}
        assert "CORRUPTION" in types
        assert "HURRICANE_RELIEF" in types

    async def test_get_by_public_id_no_entities_returns_empty_list(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article with no linked entities
        article = await _insert_article_with_text(
            db_connection,
            url="https://example.com/detail-no-entities",
            title="No entities detail article",
            full_text="Content.",
        )
        service = ArticleSearchService()

        # When: we fetch by public_id
        result = await service.get_by_public_id(db_connection, article.public_id)

        # Then: entities is an empty list, not None
        assert result is not None
        assert result.entities == []

    async def test_get_by_public_id_no_classifications_returns_empty_list(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article with no classifications
        article = await _insert_article_with_text(
            db_connection,
            url="https://example.com/detail-no-classifications",
            title="No classifications detail article",
            full_text="Content.",
        )
        service = ArticleSearchService()

        # When: we fetch by public_id
        result = await service.get_by_public_id(db_connection, article.public_id)

        # Then: classifications is an empty list, not None
        assert result is not None
        assert result.classifications == []

    async def test_get_by_public_id_populates_cache_on_miss(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article and a fresh cache
        article = await _insert_article_with_text(
            db_connection,
            url="https://example.com/detail-cache-miss",
            title="Detail cache miss article",
            full_text="Content.",
        )
        cache = InMemoryCache()
        service = ArticleSearchService(cache=cache)

        # When: we fetch by public_id for the first time
        await service.get_by_public_id(db_connection, article.public_id)

        # Then: the result is now in the cache
        assert cache.size() == 1

    async def test_get_by_public_id_returns_cached_result(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article cached by a first fetch
        article = await _insert_article_with_text(
            db_connection,
            url="https://example.com/detail-cache-hit",
            title="Detail cache hit article",
            full_text="Content.",
        )
        cache = InMemoryCache()
        service = ArticleSearchService(cache=cache)
        await service.get_by_public_id(db_connection, article.public_id)

        # And: the article is then deleted from the DB
        await delete_article(db_connection, article.id)

        # When: we fetch again
        result = await service.get_by_public_id(db_connection, article.public_id)

        # Then: the result is returned from cache (not the now-deleted DB row)
        assert result is not None
        assert result.url == "https://example.com/detail-cache-hit"

    async def test_get_by_public_id_without_cache_still_returns_result(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article and a service with no cache
        article = await _insert_article_with_text(
            db_connection,
            url="https://example.com/detail-no-cache",
            title="Detail no cache article",
            full_text="Content.",
        )
        service = ArticleSearchService(cache=None)

        # When: we fetch by public_id
        result = await service.get_by_public_id(db_connection, article.public_id)

        # Then: the article is returned normally
        assert result is not None
        assert result.url == "https://example.com/detail-no-cache"


# ---------------------------------------------------------------------------
# Related articles service tests
# ---------------------------------------------------------------------------


class TestGetRelatedArticlesService:
    """ArticleSearchService.get_related_articles() — fetch related articles by UUID."""

    async def test_returns_related_articles_without_cache(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: two articles sharing an entity, service with no cache
        source = await _insert_article_with_text(
            db_connection,
            url="https://example.com/svc-related-src",
            title="Source article",
            full_text="Content.",
        )
        other = await _insert_article_with_text(
            db_connection,
            url="https://example.com/svc-related-other",
            title="Other article",
            full_text="Content.",
        )
        entity = await create_test_entity(db_connection, name="OCG", normalized_name="ocg-svc")
        await create_test_article_entity(db_connection, source.id, entity.id)
        await create_test_article_entity(db_connection, other.id, entity.id)
        service = ArticleSearchService(cache=None)

        # When: related articles are fetched
        results = await service.get_related_articles(db_connection, source.public_id)

        # Then: the related article is returned
        assert any(r.url == "https://example.com/svc-related-other" for r in results)

    async def test_cache_miss_populates_cache(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: two articles sharing an entity and a fresh cache
        source = await _insert_article_with_text(
            db_connection,
            url="https://example.com/svc-related-cache-miss-src",
            title="Source article cache miss",
            full_text="Content.",
        )
        other = await _insert_article_with_text(
            db_connection,
            url="https://example.com/svc-related-cache-miss-other",
            title="Other article cache miss",
            full_text="Content.",
        )
        entity = await create_test_entity(db_connection, name="NHT", normalized_name="nht-svc-cache")
        await create_test_article_entity(db_connection, source.id, entity.id)
        await create_test_article_entity(db_connection, other.id, entity.id)
        cache = InMemoryCache()
        service = ArticleSearchService(cache=cache)

        # When: related articles are fetched for the first time
        await service.get_related_articles(db_connection, source.public_id)

        # Then: the result is now stored in the cache
        assert cache.size() == 1

    async def test_cache_hit_returns_without_db_call(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: two articles sharing an entity, primed cache via first fetch
        source = await _insert_article_with_text(
            db_connection,
            url="https://example.com/svc-related-cache-hit-src",
            title="Source article cache hit",
            full_text="Content.",
        )
        other = await _insert_article_with_text(
            db_connection,
            url="https://example.com/svc-related-cache-hit-other",
            title="Other article cache hit",
            full_text="Content.",
        )
        entity = await create_test_entity(db_connection, name="BOJ", normalized_name="boj-svc-hit")
        await create_test_article_entity(db_connection, source.id, entity.id)
        await create_test_article_entity(db_connection, other.id, entity.id)
        cache = InMemoryCache()
        service = ArticleSearchService(cache=cache)
        await service.get_related_articles(db_connection, source.public_id)

        # And: the related article is then deleted from the DB
        await delete_article(db_connection, other.id)

        # When: related articles are fetched again
        results = await service.get_related_articles(db_connection, source.public_id)

        # Then: the cached result is returned (not the now-deleted DB row)
        assert any(r.url == "https://example.com/svc-related-cache-hit-other" for r in results)
