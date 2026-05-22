"""Tests for ArticleRepository."""

import asyncpg
import pytest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from src.article_persistence.repositories.article_repository import ArticleRepository
from src.article_persistence.models.domain import Article
from tests.article_persistence.utils import (
    create_test_article,
    create_test_article_entity,
    create_test_classification,
    create_test_entity,
    create_test_news_source,
    insert_article_with_date,
)


class TestInsertArticleHappyPath:
    """Happy path tests for insert_article."""

    async def test_insert_article_success(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: a valid article with all fields populated
        article = Article(
            url="https://example.com/test-article",
            title="Test Article",
            section="news",
            published_date=datetime(2025, 11, 15, tzinfo=timezone.utc),
            full_text="Article content here",
            news_source_id=1,
        )
        repository = ArticleRepository()

        # When: the article is inserted
        result = await repository.insert_article(db_connection, article)

        # Then: the returned article has database-generated id and public_id, plus matching fields
        assert result.id is not None
        assert result.public_id is not None
        assert result.url == article.url
        assert result.title == article.title
        assert result.section == article.section
        assert result.published_date == article.published_date
        assert result.full_text == article.full_text
        assert result.news_source_id == 1

    async def test_insert_article_with_minimal_fields(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article with only required fields (url, title, section)
        article = Article(
            url="https://example.com/minimal-article",
            title="Minimal Article",
            section="lead-stories",
            news_source_id=1,
        )
        repository = ArticleRepository()

        # When: the article is inserted
        result = await repository.insert_article(db_connection, article)

        # Then: returns article with id, public_id, defaults applied, optional fields are None
        assert result.id is not None
        assert result.public_id is not None
        assert result.url == article.url
        assert result.title == article.title
        assert result.section == article.section
        assert result.published_date is None
        assert result.full_text is None
        assert result.fetched_at is not None


    async def test_insert_article_preserves_full_text_in_return(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article with full_text content
        full_text_content = "This is the complete article text that should be preserved."
        article = Article(
            url="https://example.com/preserve-full-text",
            title="Full Text Preservation Test",
            section="news",
            full_text=full_text_content,
            news_source_id=1,
        )
        repository = ArticleRepository()

        # When: the article is inserted
        result = await repository.insert_article(db_connection, article)

        # Then: returns article includes the original full_text
        # (verifies repository preserves it since SQL doesn't return it)
        assert result.full_text == full_text_content

    async def test_insert_article_with_http_url(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article with http:// URL (not https)
        article = Article(
            url="http://example.com/http-article",
            title="HTTP URL Article",
            section="news",
            news_source_id=1,
        )
        repository = ArticleRepository()

        # When: the article is inserted
        result = await repository.insert_article(db_connection, article)

        # Then: returns article successfully (validates http:// is accepted)
        assert result.id is not None
        assert result.url == "http://example.com/http-article"

    async def test_insert_article_strips_whitespace(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article with whitespace-padded fields
        article = Article(
            url="  https://example.com/whitespace-test  ",
            title="  Whitespace Title  ",
            section="  news  ",
            news_source_id=1,
        )
        repository = ArticleRepository()

        # When: the article is inserted
        result = await repository.insert_article(db_connection, article)

        # Then: returns article with trimmed fields (Pydantic validation)
        assert result.id is not None
        assert result.url == "https://example.com/whitespace-test"
        assert result.title == "Whitespace Title"
        assert result.section == "news"

    async def test_insert_multiple_articles_sequential(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: multiple valid articles with different URLs
        repository = ArticleRepository()
        articles = [
            Article(
                url=f"https://example.com/article-{i}",
                title=f"Article {i}",
                section="news",
                news_source_id=1,
            )
            for i in range(3)
        ]

        # When: each article is inserted sequentially
        results = []
        for article in articles:
            result = await repository.insert_article(db_connection, article)
            results.append(result)

        # Then: each gets unique auto-incrementing id
        ids = [r.id for r in results]
        assert len(set(ids)) == 3  # All IDs are unique
        assert all(id is not None for id in ids)


class TestInsertArticleDatabaseConstraints:
    """Database constraint tests for insert_article."""

    async def test_cannot_delete_news_source_with_articles(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: a news source exists with an article referencing it
        news_source = await create_test_news_source(
            conn=db_connection,
            name="News Source With Articles",
        )
        repository = ArticleRepository()
        article = Article(
            url="https://example.com/restrict-test",
            title="Article Referencing News Source",
            section="news",
            news_source_id=news_source.id,
        )
        await repository.insert_article(db_connection, article)

        # When: attempting to delete the news source
        # Then: raises ForeignKeyViolationError due to ON DELETE RESTRICT
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            await db_connection.execute(
                "DELETE FROM news_sources WHERE id = $1",
                news_source.id,
            )

    async def test_duplicate_url_raises_unique_violation(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article already exists with a specific URL
        repository = ArticleRepository()
        first_article = Article(
            url="https://example.com/duplicate-test",
            title="First Article",
            section="news",
            news_source_id=1,
        )
        await repository.insert_article(db_connection, first_article)

        # When: another article with the same URL is inserted
        second_article = Article(
            url="https://example.com/duplicate-test",
            title="Second Article",
            section="lead-stories",
            news_source_id=1,
        )

        # Then: raises asyncpg.UniqueViolationError
        with pytest.raises(asyncpg.UniqueViolationError):
            await repository.insert_article(db_connection, second_article)

    async def test_same_url_different_section_fails(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article exists with URL X
        repository = ArticleRepository()
        first_article = Article(
            url="https://example.com/url-unique-test",
            title="News Article",
            section="news",
            news_source_id=1,
        )
        await repository.insert_article(db_connection, first_article)

        # When: another article with same URL X but different section is inserted
        second_article = Article(
            url="https://example.com/url-unique-test",
            title="Lead Story Article",
            section="lead-stories",
            news_source_id=1,
        )

        # Then: raises UniqueViolationError (URL uniqueness is global, not per-section)
        with pytest.raises(asyncpg.UniqueViolationError):
            await repository.insert_article(db_connection, second_article)


class TestInsertArticleEdgeCases:
    """Edge case tests for insert_article."""

    async def test_with_special_characters_in_url(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article with URL containing query params, fragments, encoded characters
        special_url = "https://example.com/article?param=value&other=123#section-1"
        article = Article(
            url=special_url,
            title="Special URL Article",
            section="news",
            news_source_id=1,
        )
        repository = ArticleRepository()

        # When: the article is inserted
        result = await repository.insert_article(db_connection, article)

        # Then: returns article with URL preserved correctly
        assert result.id is not None
        assert result.url == special_url

    async def test_with_unicode_title(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article with Unicode characters in title (accents, special chars)
        unicode_title = "Café Culture: Jamaica's Growing Artisanal Scene — 日本語テスト"
        article = Article(
            url="https://example.com/unicode-title",
            title=unicode_title,
            section="news",
            news_source_id=1,
        )
        repository = ArticleRepository()

        # When: the article is inserted
        result = await repository.insert_article(db_connection, article)

        # Then: returns article with Unicode title preserved
        assert result.id is not None
        assert result.title == unicode_title

    async def test_with_very_long_full_text(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article with very long full_text (100KB of text)
        long_text = "This is a test paragraph. " * 5000  # ~130KB
        article = Article(
            url="https://example.com/long-full-text",
            title="Long Content Article",
            section="news",
            full_text=long_text,
            news_source_id=1,
        )
        repository = ArticleRepository()

        # When: the article is inserted
        result = await repository.insert_article(db_connection, article)

        # Then: successfully inserts (TEXT type handles large content)
        assert result.id is not None
        assert result.full_text == long_text

    async def test_fetched_at_defaults_to_current_time(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article without explicit fetched_at
        before_insert = datetime.now(timezone.utc)
        article = Article(
            url="https://example.com/default-fetched-at",
            title="Default Fetched At Article",
            section="news",
            news_source_id=1,
        )
        repository = ArticleRepository()

        # When: the article is inserted
        result = await repository.insert_article(db_connection, article)
        after_insert = datetime.now(timezone.utc)

        # Then: returns article with fetched_at close to current time
        assert result.id is not None
        assert result.fetched_at is not None
        # Allow 1 second tolerance for test execution time
        assert before_insert - timedelta(seconds=1) <= result.fetched_at <= after_insert + timedelta(seconds=1)

    async def test_with_custom_fetched_at(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article with explicit fetched_at value
        custom_fetched_at = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        article = Article(
            url="https://example.com/custom-fetched-at",
            title="Custom Fetched At Article",
            section="news",
            fetched_at=custom_fetched_at,
            news_source_id=1,
        )
        repository = ArticleRepository()

        # When: the article is inserted
        result = await repository.insert_article(db_connection, article)

        # Then: returns article with the custom fetched_at preserved
        assert result.id is not None
        assert result.fetched_at == custom_fetched_at


class TestGetByPublicId:
    """Tests for get_by_public_id method."""

    async def test_get_by_public_id_success(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article exists in the database
        article = Article(
            url="https://example.com/public-id-lookup",
            title="Public ID Lookup Test",
            section="news",
            full_text="Content for public ID test",
            news_source_id=1,
        )
        repository = ArticleRepository()
        inserted = await repository.insert_article(db_connection, article)

        # When: the article is retrieved by its public_id
        result = await repository.get_by_public_id(db_connection, inserted.public_id)

        # Then: returns the same article with all fields populated
        assert result is not None
        assert result.id == inserted.id
        assert result.public_id == inserted.public_id
        assert result.url == article.url
        assert result.title == article.title
        assert result.section == article.section
        assert result.full_text == article.full_text

    async def test_get_by_public_id_not_found(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: a random UUID that doesn't exist in the database
        non_existent_uuid = uuid4()
        repository = ArticleRepository()

        # When: attempting to retrieve by that UUID
        result = await repository.get_by_public_id(db_connection, non_existent_uuid)

        # Then: returns None
        assert result is None

    async def test_public_id_is_unique_across_articles(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: multiple articles are inserted
        repository = ArticleRepository()
        articles = [
            Article(
                url=f"https://example.com/unique-public-id-{i}",
                title=f"Unique Public ID Test {i}",
                section="news",
                news_source_id=1,
            )
            for i in range(3)
        ]

        # When: each article is inserted
        results = []
        for article in articles:
            result = await repository.insert_article(db_connection, article)
            results.append(result)

        # Then: each article has a unique public_id
        public_ids = [r.public_id for r in results]
        assert len(set(public_ids)) == 3  # All public_ids are unique
        assert all(pid is not None for pid in public_ids)


class TestGetRelatedArticlesHappyPath:
    """Happy path tests for get_related_articles_by_public_id."""

    async def test_two_articles_sharing_one_entity_returns_related(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: two articles linked to the same entity
        source = await create_test_article(db_connection, url="https://example.com/related-src-1", news_source_id=1)
        other = await create_test_article(db_connection, url="https://example.com/related-other-1", news_source_id=1)
        entity = await create_test_entity(db_connection, name="INDECOM", normalized_name="indecom-rel-1")
        await create_test_article_entity(db_connection, source.id, entity.id)
        await create_test_article_entity(db_connection, other.id, entity.id)
        repo = ArticleRepository()

        # When: related articles are fetched for the source article
        results = await repo.get_related_articles_by_public_id(db_connection, source.public_id)

        # Then: the other article appears in the results
        result_ids = [r.public_id for r in results]
        assert other.public_id in result_ids
        assert source.public_id not in result_ids

    async def test_ranking_by_shared_entity_count(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: source article shares 2 entities with article A and 1 entity with article B
        source = await create_test_article(db_connection, url="https://example.com/rank-src", news_source_id=1)
        article_a = await create_test_article(db_connection, url="https://example.com/rank-a", news_source_id=1)
        article_b = await create_test_article(db_connection, url="https://example.com/rank-b", news_source_id=1)

        entity1 = await create_test_entity(db_connection, name="Petrojam", normalized_name="petrojam-rank")
        entity2 = await create_test_entity(db_connection, name="NWC", normalized_name="nwc-rank")

        # Link source and article_a to both entities
        await create_test_article_entity(db_connection, source.id, entity1.id)
        await create_test_article_entity(db_connection, source.id, entity2.id)
        await create_test_article_entity(db_connection, article_a.id, entity1.id)
        await create_test_article_entity(db_connection, article_a.id, entity2.id)
        # Link source and article_b to only one entity
        await create_test_article_entity(db_connection, article_b.id, entity1.id)

        repo = ArticleRepository()

        # When: related articles are fetched
        results = await repo.get_related_articles_by_public_id(db_connection, source.public_id)

        # Then: article_a (2 shared entities) ranks above article_b (1 shared entity)
        result_ids = [r.public_id for r in results]
        assert result_ids.index(article_a.public_id) < result_ids.index(article_b.public_id)

    async def test_related_articles_come_from_multiple_news_sources(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: a second news source distinct from the default (id=1)
        second_source = await create_test_news_source(
            db_connection,
            name="Test Observer Multi Source",
            base_url="https://test-observer-multi.com",
        )

        source = await create_test_article(db_connection, url="https://example.com/multi-src-source", news_source_id=1)
        gleaner_article = await create_test_article(db_connection, url="https://example.com/multi-src-gleaner", news_source_id=1)
        observer_article = await create_test_article(db_connection, url="https://example.com/multi-src-observer", news_source_id=second_source.id)

        entity = await create_test_entity(db_connection, name="Parliament", normalized_name="parliament-multi")
        await create_test_article_entity(db_connection, source.id, entity.id)
        await create_test_article_entity(db_connection, gleaner_article.id, entity.id)
        await create_test_article_entity(db_connection, observer_article.id, entity.id)

        repo = ArticleRepository()

        # When: related articles are fetched for the source article
        results = await repo.get_related_articles_by_public_id(db_connection, source.public_id)

        # Then: related articles from both news sources are returned
        result_ids = {r.public_id for r in results}
        assert gleaner_article.public_id in result_ids
        assert observer_article.public_id in result_ids

    async def test_article_with_no_entities_returns_empty_list(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: an article with no entity links
        source = await create_test_article(db_connection, url="https://example.com/no-entities", news_source_id=1)
        repo = ArticleRepository()

        # When: related articles are fetched
        results = await repo.get_related_articles_by_public_id(db_connection, source.public_id)

        # Then: empty list returned
        assert results == []

    async def test_no_shared_entities_between_two_articles_returns_empty_list(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: two articles each linked to a different entity (no overlap)
        source = await create_test_article(db_connection, url="https://example.com/no-shared-src", news_source_id=1)
        other = await create_test_article(db_connection, url="https://example.com/no-shared-other", news_source_id=1)
        entity_a = await create_test_entity(db_connection, name="Body A", normalized_name="body-a-no-shared")
        entity_b = await create_test_entity(db_connection, name="Body B", normalized_name="body-b-no-shared")
        await create_test_article_entity(db_connection, source.id, entity_a.id)
        await create_test_article_entity(db_connection, other.id, entity_b.id)
        repo = ArticleRepository()

        # When: related articles are fetched for the source article
        results = await repo.get_related_articles_by_public_id(db_connection, source.public_id)

        # Then: empty list returned since no entities are shared
        assert results == []

    async def test_limit_parameter_caps_results(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: source article shares an entity with 6 other articles
        source = await create_test_article(db_connection, url="https://example.com/limit-src", news_source_id=1)
        entity = await create_test_entity(db_connection, name="GOJ", normalized_name="goj-limit")
        await create_test_article_entity(db_connection, source.id, entity.id)

        for i in range(6):
            art = await create_test_article(db_connection, url=f"https://example.com/limit-art-{i}", news_source_id=1)
            await create_test_article_entity(db_connection, art.id, entity.id)

        repo = ArticleRepository()

        # When: related articles are fetched with limit=5
        results = await repo.get_related_articles_by_public_id(db_connection, source.public_id, limit=5)

        # Then: at most 5 results returned
        assert len(results) == 5


class TestGetRelatedArticlesOrdering:
    """Tests that verify the confidence → date → entity count sort order."""

    async def test_entity_count_sorts_before_confidence(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given:
        #   article_a: 2 shared entities, low confidence (0.5)
        #   article_b: 1 shared entity,  high confidence (0.9), newer date
        #   article_c: 1 shared entity,  low confidence (0.5), older date
        older = datetime(2024, 1, 1, tzinfo=timezone.utc)
        newer = datetime(2024, 6, 1, tzinfo=timezone.utc)

        source = await insert_article_with_date(db_connection, url="https://example.com/ord-src", published_date=older)
        article_a = await insert_article_with_date(db_connection, url="https://example.com/ord-a", published_date=older)
        article_b = await insert_article_with_date(db_connection, url="https://example.com/ord-b", published_date=newer)
        article_c = await insert_article_with_date(db_connection, url="https://example.com/ord-c", published_date=older)

        entity1 = await create_test_entity(db_connection, name="Petrojam Ord", normalized_name="petrojam-ord")
        entity2 = await create_test_entity(db_connection, name="NWC Ord", normalized_name="nwc-ord")

        # source and article_a share both entities; article_b and article_c share only entity1
        for art in [source, article_a]:
            await create_test_article_entity(db_connection, art.id, entity1.id)
            await create_test_article_entity(db_connection, art.id, entity2.id)
        for art in [article_b, article_c]:
            await create_test_article_entity(db_connection, art.id, entity1.id)

        await create_test_classification(db_connection, article_a.id, confidence_score=0.5)
        await create_test_classification(db_connection, article_b.id, confidence_score=0.9)
        await create_test_classification(db_connection, article_c.id, confidence_score=0.5)

        repo = ArticleRepository()

        # When: related articles are fetched
        results = await repo.get_related_articles_by_public_id(db_connection, source.public_id, limit=10)

        # Then: order is A (2 entities) → B (1 entity, high conf) → C (1 entity, low conf, older)
        result_ids = [r.public_id for r in results]
        assert result_ids.index(article_a.public_id) < result_ids.index(article_b.public_id)
        assert result_ids.index(article_b.public_id) < result_ids.index(article_c.public_id)

    async def test_unclassified_articles_sort_last(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: two related articles — one classified (confidence 0.7), one unclassified
        source = await create_test_article(db_connection, url="https://example.com/ord-unclass-src", news_source_id=1)
        classified = await create_test_article(db_connection, url="https://example.com/ord-unclass-yes", news_source_id=1)
        unclassified = await create_test_article(db_connection, url="https://example.com/ord-unclass-no", news_source_id=1)

        entity = await create_test_entity(db_connection, name="NSWMA Ord", normalized_name="nswma-ord")
        for art in [source, classified, unclassified]:
            await create_test_article_entity(db_connection, art.id, entity.id)

        await create_test_classification(db_connection, classified.id, confidence_score=0.7)

        repo = ArticleRepository()

        # When: related articles are fetched
        results = await repo.get_related_articles_by_public_id(db_connection, source.public_id, limit=10)

        # Then: classified article appears before the unclassified one
        result_ids = [r.public_id for r in results]
        assert result_ids.index(classified.public_id) < result_ids.index(unclassified.public_id)


class TestGetRelatedArticlesEdgeCases:
    """Edge case tests for get_related_articles_by_public_id."""

    async def test_nonexistent_public_id_returns_empty_list(
        self,
        db_connection: asyncpg.Connection,
    ):
        # Given: a UUID that does not exist in the database
        nonexistent_id = uuid4()
        repo = ArticleRepository()

        # When: related articles are fetched for the nonexistent id
        results = await repo.get_related_articles_by_public_id(db_connection, nonexistent_id)

        # Then: empty list returned (not an error)
        assert results == []
