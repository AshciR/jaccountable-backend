"""Tests for the analytics event fired by the search_articles route.

These tests verify that search:query_submit is captured with the correct
properties when GET /api/v1/articles is called.

Strategy:
- Use FastAPI TestClient for HTTP-level testing
- Mock db_config pool lifecycle so no real database is needed
- Override get_db and get_analytics dependencies
- Patch ArticleSearchService.search to return controlled (results, total) tuples
- Assert on mock_analytics.capture_with_common_props call arguments
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest
from fastapi.testclient import TestClient

from config.database import db_config
from src.analytics.client import AnalyticsClient
from src.server.app import app
from src.server.articles.service import ArticleSearchService
from src.server.dependencies import get_analytics, get_db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_analytics() -> MagicMock:
    """Spy on AnalyticsClient without making real PostHog HTTP requests."""
    mock = MagicMock(spec=AnalyticsClient)
    mock.disabled = False
    mock.environment = "test"
    mock.get_distinct_id.return_value = "test-distinct-id"
    mock.is_internal_request.return_value = False
    mock.get_session_id.return_value = None
    return mock


@pytest.fixture
def client(mock_analytics: MagicMock) -> TestClient:
    """TestClient with database and analytics dependencies overridden.

    Patches db_config pool lifecycle methods so no real PostgreSQL connection
    is required — these tests focus purely on analytics behaviour.
    """

    async def override_get_db():
        yield MagicMock(spec=asyncpg.Connection)

    def override_get_analytics() -> AnalyticsClient:
        return mock_analytics

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_analytics] = override_get_analytics

    with (
        patch.object(db_config, "create_pool", new_callable=AsyncMock),
        patch.object(db_config, "close_pool", new_callable=AsyncMock),
    ):
        with TestClient(app) as c:
            yield c

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# TestSearchAnalyticsEventFired
# ---------------------------------------------------------------------------


class TestSearchAnalyticsEventFired:
    """search:query_submit is captured for every successful search request."""

    async def test_event_fired_with_search_query_and_results_count(
        self, client: TestClient, mock_analytics: MagicMock
    ):
        # Given: service returns 3 results for a query
        with patch.object(
            ArticleSearchService, "search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = ([], 3)

            # When: client submits a search with q=corruption
            response = client.get("/api/v1/articles", params={"q": "corruption"})

        # Then: HTTP succeeds and event was captured
        assert response.status_code == 200
        mock_analytics.capture_with_common_props.assert_called_once()
        call_kwargs = mock_analytics.capture_with_common_props.call_args.kwargs
        assert call_kwargs["event"] == "search:query_submit"
        assert call_kwargs["properties"]["search_query"] == "corruption"
        assert call_kwargs["properties"]["results_count"] == 3

    async def test_event_fired_with_none_query_in_browse_mode(
        self, client: TestClient, mock_analytics: MagicMock
    ):
        # Given: service returns articles for a browse (no-query) request
        with patch.object(
            ArticleSearchService, "search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = ([], 42)

            # When: client calls with no q param (browse mode)
            response = client.get("/api/v1/articles")

        # Then: search_query is None in the event (browse mode, not a bug)
        assert response.status_code == 200
        call_kwargs = mock_analytics.capture_with_common_props.call_args.kwargs
        assert call_kwargs["properties"]["search_query"] is None
        assert call_kwargs["properties"]["results_count"] == 42

    async def test_results_count_reflects_total_not_page_size(
        self, client: TestClient, mock_analytics: MagicMock
    ):
        # Given: service returns total=100 with only 20 items on this page
        with patch.object(
            ArticleSearchService, "search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = ([], 100)

            # When: client fetches page 1 with page_size=20
            client.get("/api/v1/articles", params={"page": 1, "page_size": 20})

        # Then: results_count is the total count, not the page size
        call_kwargs = mock_analytics.capture_with_common_props.call_args.kwargs
        assert call_kwargs["properties"]["results_count"] == 100

    async def test_distinct_id_and_is_internal_come_from_analytics_helpers(
        self, client: TestClient, mock_analytics: MagicMock
    ):
        # Given: analytics helpers return controlled values
        mock_analytics.get_distinct_id.return_value = "frontend-posthog-id"
        mock_analytics.is_internal_request.return_value = True

        with patch.object(
            ArticleSearchService, "search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = ([], 0)

            # When: a request is made
            client.get("/api/v1/articles")

        # Then: the router passes those values to capture_with_common_props
        call_kwargs = mock_analytics.capture_with_common_props.call_args.kwargs
        assert call_kwargs["distinct_id"] == "frontend-posthog-id"
        assert call_kwargs["is_internal"] is True

    async def test_query_submit_not_fired_when_page_greater_than_one(
        self, client: TestClient, mock_analytics: MagicMock
    ):
        # Given: service returns results for a page-2 request
        with patch.object(
            ArticleSearchService, "search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = ([], 10)

            # When: client fetches page 2
            client.get("/api/v1/articles", params={"page": 2})

        # Then: search:query_submit was NOT fired
        call_kwargs = mock_analytics.capture_with_common_props.call_args.kwargs
        assert call_kwargs["event"] != "search:query_submit"


# ---------------------------------------------------------------------------
# TestLoadMoreAnalyticsEventFired
# ---------------------------------------------------------------------------


class TestLoadMoreAnalyticsEventFired:
    """search:load_more_click is captured when page > 1."""

    async def test_event_fired_with_correct_properties_on_page_two(
        self, client: TestClient, mock_analytics: MagicMock
    ):
        # Given: service returns results for a page-2 search
        with patch.object(
            ArticleSearchService, "search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = ([], 20)

            # When: client fetches page 2 with a query
            client.get(
                "/api/v1/articles", params={"q": "corruption", "page": 2, "page_size": 5}
            )

        # Then: search:load_more_click is captured with correct properties
        mock_analytics.capture_with_common_props.assert_called_once()
        call_kwargs = mock_analytics.capture_with_common_props.call_args.kwargs
        assert call_kwargs["event"] == "search:load_more_click"
        assert call_kwargs["properties"]["search_query"] == "corruption"
        assert call_kwargs["properties"]["page"] == 2
        assert call_kwargs["properties"]["current_results_count"] == 5  # (2-1) * 5

    async def test_search_query_is_empty_string_in_browse_mode(
        self, client: TestClient, mock_analytics: MagicMock
    ):
        # Given: service returns results for a browse (no-query) load-more request
        with patch.object(
            ArticleSearchService, "search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = ([], 50)

            # When: client fetches page 2 with no q param
            client.get("/api/v1/articles", params={"page": 2})

        # Then: search_query is "" (not None) to match frontend spec
        call_kwargs = mock_analytics.capture_with_common_props.call_args.kwargs
        assert call_kwargs["properties"]["search_query"] == ""

    async def test_current_results_count_is_previous_pages_times_page_size(
        self, client: TestClient, mock_analytics: MagicMock
    ):
        # Given: service returns results for page 3 with page_size=5
        with patch.object(
            ArticleSearchService, "search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = ([], 30)

            # When: client fetches page 3
            client.get("/api/v1/articles", params={"page": 3, "page_size": 5})

        # Then: current_results_count = (3-1) * 5 = 10
        call_kwargs = mock_analytics.capture_with_common_props.call_args.kwargs
        assert call_kwargs["properties"]["current_results_count"] == 10

    async def test_query_submit_not_fired_on_load_more(
        self, client: TestClient, mock_analytics: MagicMock
    ):
        # Given: service returns results for a page-2 request
        with patch.object(
            ArticleSearchService, "search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = ([], 10)

            # When: client fetches page 2
            client.get("/api/v1/articles", params={"page": 2})

        # Then: search:query_submit was NOT fired
        call_kwargs = mock_analytics.capture_with_common_props.call_args.kwargs
        assert call_kwargs["event"] != "search:query_submit"


# ---------------------------------------------------------------------------
# TestSearchAnalyticsInternalFlag
# ---------------------------------------------------------------------------


class TestSearchAnalyticsInternalFlag:
    """is_internal is sourced from analytics.is_internal_request(request)."""

    async def test_is_internal_true_when_helper_returns_true(
        self, client: TestClient, mock_analytics: MagicMock
    ):
        # Given: is_internal_request returns True
        mock_analytics.is_internal_request.return_value = True

        with patch.object(
            ArticleSearchService, "search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = ([], 0)

            # When: request is made
            client.get("/api/v1/articles", headers={"X-Internal-Request": "true"})

        # Then: is_internal=True is passed to capture_with_common_props
        call_kwargs = mock_analytics.capture_with_common_props.call_args.kwargs
        assert call_kwargs["is_internal"] is True

    async def test_is_internal_false_when_helper_returns_false(
        self, client: TestClient, mock_analytics: MagicMock
    ):
        # Given: is_internal_request returns False (default)
        mock_analytics.is_internal_request.return_value = False

        with patch.object(
            ArticleSearchService, "search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = ([], 0)

            # When: request has no internal header
            client.get("/api/v1/articles")

        # Then: is_internal=False is passed to capture_with_common_props
        call_kwargs = mock_analytics.capture_with_common_props.call_args.kwargs
        assert call_kwargs["is_internal"] is False


class TestSearchAnalyticsSessionId:
    """session_id is sourced from analytics.get_session_id(request)."""

    async def test_session_id_propagated_when_helper_returns_value(
        self, client: TestClient, mock_analytics: MagicMock
    ):
        # Given: get_session_id returns a session id (header was present)
        mock_analytics.get_session_id.return_value = "session-search-abc"

        with patch.object(
            ArticleSearchService, "search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = ([], 0)

            # When: request is made with the PostHog session header
            client.get(
                "/api/v1/articles",
                headers={"X-PostHog-Session-Id": "session-search-abc"},
            )

        # Then: session_id is passed through to capture_with_common_props
        call_kwargs = mock_analytics.capture_with_common_props.call_args.kwargs
        assert call_kwargs["session_id"] == "session-search-abc"

    async def test_session_id_none_when_helper_returns_none(
        self, client: TestClient, mock_analytics: MagicMock
    ):
        # Given: get_session_id returns None (header was absent)
        mock_analytics.get_session_id.return_value = None

        with patch.object(
            ArticleSearchService, "search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = ([], 0)

            # When: request is made without the PostHog session header
            client.get("/api/v1/articles")

        # Then: session_id=None is passed (client method will then omit $session_id)
        call_kwargs = mock_analytics.capture_with_common_props.call_args.kwargs
        assert call_kwargs["session_id"] is None


# ---------------------------------------------------------------------------
# TestArticleDetailAnalytics
# ---------------------------------------------------------------------------


def _make_detail_result(
    public_id: str = "11111111-1111-1111-1111-111111111111",
    title: str = "Public Funds Misused",
    section: str = "lead-stories",
) -> SimpleNamespace:
    """Build a minimal ArticleSearchResult-like object for ArticleSearchResultSchema.model_validate."""
    return SimpleNamespace(
        public_id=public_id,
        url="https://example.com/article",
        title=title,
        section=section,
        published_date=None,
        news_source_id=1,
        snippet=None,
        entities=[],
        classifications=[],
        full_text=None,
    )


class TestArticleDetailAnalytics:
    """article:detail_view is captured on successful GET /articles/{public_id}."""

    _PUBLIC_ID = "11111111-1111-1111-1111-111111111111"

    async def test_event_fired_with_x_source_header(
        self, client: TestClient, mock_analytics: MagicMock
    ):
        # Given: service returns an article and the request includes X-Source
        mock_analytics.get_navigation_source.return_value = "home"

        with patch.object(
            ArticleSearchService, "get_by_public_id", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = _make_detail_result(
                public_id=self._PUBLIC_ID,
                title="Public Funds Misused",
                section="lead-stories",
            )

            # When: client fetches the article with X-Source: home
            response = client.get(
                f"/api/v1/articles/{self._PUBLIC_ID}",
                headers={"X-Source": "home"},
            )

        # Then: HTTP succeeds and the detail-view event is captured
        assert response.status_code == 200
        mock_analytics.capture_with_common_props.assert_called_once()
        call_kwargs = mock_analytics.capture_with_common_props.call_args.kwargs
        assert call_kwargs["event"] == "article:detail_view"
        assert call_kwargs["properties"]["article_public_id"] == self._PUBLIC_ID
        assert call_kwargs["properties"]["article_title"] == "Public Funds Misused"
        assert call_kwargs["properties"]["article_section"] == "lead-stories"
        assert call_kwargs["properties"]["navigation_source"] == "home"

    async def test_navigation_source_sourced_from_analytics_helper(
        self, client: TestClient, mock_analytics: MagicMock
    ):
        # Given: get_navigation_source returns a controlled value
        mock_analytics.get_navigation_source.return_value = "unknown"

        with patch.object(
            ArticleSearchService, "get_by_public_id", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = _make_detail_result(public_id=self._PUBLIC_ID)

            # When: client fetches without X-Source
            client.get(f"/api/v1/articles/{self._PUBLIC_ID}")

        # Then: the helper's return value is passed through as navigation_source
        call_kwargs = mock_analytics.capture_with_common_props.call_args.kwargs
        assert call_kwargs["properties"]["navigation_source"] == "unknown"

    async def test_event_not_fired_on_404(
        self, client: TestClient, mock_analytics: MagicMock
    ):
        # Given: service returns None (article not found)
        with patch.object(
            ArticleSearchService, "get_by_public_id", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = None

            # When: client fetches a non-existent article
            response = client.get(f"/api/v1/articles/{self._PUBLIC_ID}")

        # Then: HTTP 404 AND no analytics event is captured
        assert response.status_code == 404
        mock_analytics.capture_with_common_props.assert_not_called()

    async def test_distinct_id_and_is_internal_propagated(
        self, client: TestClient, mock_analytics: MagicMock
    ):
        # Given: analytics helpers return controlled values
        mock_analytics.get_distinct_id.return_value = "frontend-posthog-id"
        mock_analytics.is_internal_request.return_value = True
        mock_analytics.get_navigation_source.return_value = "home"

        with patch.object(
            ArticleSearchService, "get_by_public_id", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = _make_detail_result(public_id=self._PUBLIC_ID)

            # When: a detail request is made
            client.get(f"/api/v1/articles/{self._PUBLIC_ID}")

        # Then: distinct_id and is_internal are propagated to the event
        call_kwargs = mock_analytics.capture_with_common_props.call_args.kwargs
        assert call_kwargs["distinct_id"] == "frontend-posthog-id"
        assert call_kwargs["is_internal"] is True

    async def test_session_id_propagated_when_helper_returns_value(
        self, client: TestClient, mock_analytics: MagicMock
    ):
        # Given: get_session_id returns a value (X-PostHog-Session-Id was present)
        mock_analytics.get_session_id.return_value = "session-detail-xyz"

        with patch.object(
            ArticleSearchService, "get_by_public_id", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = _make_detail_result(public_id=self._PUBLIC_ID)

            # When: detail request is made with the PostHog session header
            client.get(
                f"/api/v1/articles/{self._PUBLIC_ID}",
                headers={"X-PostHog-Session-Id": "session-detail-xyz"},
            )

        # Then: session_id is forwarded to capture_with_common_props
        call_kwargs = mock_analytics.capture_with_common_props.call_args.kwargs
        assert call_kwargs["session_id"] == "session-detail-xyz"

    async def test_session_id_none_when_helper_returns_none(
        self, client: TestClient, mock_analytics: MagicMock
    ):
        # Given: get_session_id returns None (no PostHog session header on request)
        mock_analytics.get_session_id.return_value = None

        with patch.object(
            ArticleSearchService, "get_by_public_id", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = _make_detail_result(public_id=self._PUBLIC_ID)

            # When: detail request is made without the session header
            client.get(f"/api/v1/articles/{self._PUBLIC_ID}")

        # Then: session_id=None is forwarded so client omits $session_id from event
        call_kwargs = mock_analytics.capture_with_common_props.call_args.kwargs
        assert call_kwargs["session_id"] is None

