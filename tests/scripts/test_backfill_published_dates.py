"""Tests for the Gleaner published-date backfill script."""
from datetime import datetime, timezone

import asyncpg

from scripts.production.classification.backfill_published_dates import (
    backfill_published_dates,
    parse_gleaner_url_date,
)
from tests.article_persistence.utils import (
    create_test_article,
    create_test_news_source,
    insert_article_with_date,
)

GLEANER_NEWS_SOURCE_ID = 1  # seeded by migration


async def get_published_date(conn: asyncpg.Connection, url: str) -> datetime | None:
    return await conn.fetchval("SELECT published_date FROM articles WHERE url = $1", url)


class TestParseGleanerUrlDate:
    """URL date parsing."""

    async def test_parses_date_from_article_url(self):
        # Given: a Gleaner article URL with a YYYYMMDD segment
        url = "http://jamaica-gleaner.com/article/news/20260925/mining-ministry-moves-overhaul-regulation-after-audit-findings"

        # When: parsing the date
        result = parse_gleaner_url_date(url)

        # Then: midnight UTC on that day
        assert result == datetime(2026, 9, 25, tzinfo=timezone.utc)

    async def test_url_without_date_returns_none(self):
        # Given: an archive URL without the /article/{section}/{YYYYMMDD}/ pattern
        url = "https://gleaner.newspaperarchive.com/kingston-gleaner/2021-11-01/page-3/"

        # When / Then: no date
        assert parse_gleaner_url_date(url) is None

    async def test_invalid_calendar_date_returns_none(self):
        # Given: a URL with an impossible date
        url = "https://jamaica-gleaner.com/article/news/20260231/bad-date"

        # When / Then: no date
        assert parse_gleaner_url_date(url) is None


class TestBackfillPublishedDates:
    """Backfill against a real database."""

    async def test_fills_null_dates_from_url(self, db_connection: asyncpg.Connection):
        # Given: a Gleaner article with a NULL published_date
        url = "https://jamaica-gleaner.com/article/news/20260925/backfill-test"
        await create_test_article(db_connection, url=url, news_source_id=GLEANER_NEWS_SOURCE_ID)

        # When: running the backfill
        summary = await backfill_published_dates(db_connection)

        # Then: the date is set from the URL
        assert summary.updated == 1
        assert await get_published_date(db_connection, url) == datetime(2026, 9, 25, tzinfo=timezone.utc)

    async def test_dry_run_does_not_persist(self, db_connection: asyncpg.Connection):
        # Given: a Gleaner article with a NULL published_date
        url = "https://jamaica-gleaner.com/article/news/20260925/dry-run-test"
        await create_test_article(db_connection, url=url, news_source_id=GLEANER_NEWS_SOURCE_ID)

        # When: running the backfill in dry-run mode
        summary = await backfill_published_dates(db_connection, dry_run=True)

        # Then: the update is reported but rolled back
        assert summary.updated == 1
        assert await get_published_date(db_connection, url) is None

    async def test_existing_dates_untouched(self, db_connection: asyncpg.Connection):
        # Given: a Gleaner article that already has a date different from its URL
        url = "https://jamaica-gleaner.com/article/news/20260925/has-date"
        existing = datetime(2026, 9, 25, 5, 6, 7, tzinfo=timezone.utc)
        await insert_article_with_date(db_connection, url=url, published_date=existing)

        # When: running the backfill
        summary = await backfill_published_dates(db_connection)

        # Then: it is not a candidate and keeps its exact timestamp
        assert summary.candidates == 0
        assert await get_published_date(db_connection, url) == existing

    async def test_other_news_sources_ignored(self, db_connection: asyncpg.Connection):
        # Given: a non-Gleaner article with a NULL date and a Gleaner-like URL
        source = await create_test_news_source(db_connection, name="Other Source", base_url="https://other.com")
        url = "https://other.com/article/news/20260925/not-gleaner"
        await create_test_article(db_connection, url=url, news_source_id=source.id)

        # When: running the backfill
        summary = await backfill_published_dates(db_connection)

        # Then: it is left alone
        assert summary.candidates == 0
        assert await get_published_date(db_connection, url) is None

    async def test_url_without_date_reported_and_left_null(self, db_connection: asyncpg.Connection):
        # Given: a Gleaner article whose URL has no date segment
        url = "https://jamaica-gleaner.com/some/other/path"
        await create_test_article(db_connection, url=url, news_source_id=GLEANER_NEWS_SOURCE_ID)

        # When: running the backfill
        summary = await backfill_published_dates(db_connection)

        # Then: it is reported as unparseable and stays NULL
        assert summary.unparseable_urls == [url]
        assert summary.updated == 0
        assert await get_published_date(db_connection, url) is None

    async def test_limit_caps_candidates(self, db_connection: asyncpg.Connection):
        # Given: three Gleaner articles with NULL dates
        for day in ("01", "02", "03"):
            await create_test_article(
                db_connection,
                url=f"https://jamaica-gleaner.com/article/news/202609{day}/limit-{day}",
                news_source_id=GLEANER_NEWS_SOURCE_ID,
            )

        # When: running with limit=2
        summary = await backfill_published_dates(db_connection, limit=2)

        # Then: only two are processed
        assert summary.candidates == 2
        assert summary.updated == 2
