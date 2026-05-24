"""Unit tests for AnalyticsClient."""

from unittest.mock import MagicMock, patch

import pytest

from src.analytics.client import AnalyticsClient


class TestAnalyticsClientDisabled:
    """Client is in no-op mode when POSTHOG_API_KEY is absent or empty."""

    async def test_disabled_when_no_api_key(self):
        # Given: no API key provided
        # When: client is created with empty key
        client = AnalyticsClient(api_key="")

        # Then: client reports disabled
        assert client.disabled is True

    async def test_capture_does_not_raise_when_disabled(self):
        # Given: a disabled client
        client = AnalyticsClient(api_key="")

        # When / Then: calling _capture does not raise
        client._capture(distinct_id="test", event="test:event", properties={})

    async def test_capture_with_common_props_does_not_raise_when_disabled(self):
        # Given: a disabled client
        client = AnalyticsClient(api_key="")

        # When / Then: calling capture_with_common_props does not raise
        client.capture_with_common_props(
            distinct_id="test",
            event="test:event",
            properties={"key": "value"},
            is_internal=False,
        )

    async def test_shutdown_does_not_raise_when_disabled(self):
        # Given: a disabled client
        client = AnalyticsClient(api_key="")

        # When / Then: shutdown does not raise
        client.shutdown()


class TestAnalyticsClientEnabled:
    """Client routes events to the PostHog SDK when API key is set."""

    async def test_capture_calls_posthog_capture(self):
        # Given: a client with an API key
        with patch("src.analytics.client.posthog.Posthog") as MockPosthog:
            mock_ph = MagicMock()
            MockPosthog.return_value = mock_ph
            client = AnalyticsClient(api_key="phc_test_key_123")

            # When: _capture is called
            client._capture(
                distinct_id="user-abc",
                event="search:query_submit",
                properties={"search_query": "corruption"},
            )

        # Then: the underlying posthog client was invoked
        mock_ph.capture.assert_called_once_with(
            distinct_id="user-abc",
            event="search:query_submit",
            properties={"search_query": "corruption"},
        )

    async def test_capture_with_common_props_merges_environment_and_is_internal(self):
        # Given: a client configured for staging
        with patch("src.analytics.client.posthog.Posthog") as MockPosthog:
            mock_ph = MagicMock()
            MockPosthog.return_value = mock_ph
            client = AnalyticsClient(api_key="phc_test_key_123", environment="staging")

            # When: capture_with_common_props is called with event-specific props
            client.capture_with_common_props(
                distinct_id="user-xyz",
                event="search:query_submit",
                properties={"search_query": "budget", "results_count": 5},
                is_internal=True,
            )

        # Then: common props are merged alongside event-specific ones
        props = mock_ph.capture.call_args.kwargs["properties"]
        assert props["environment"] == "staging"
        assert props["is_internal"] is True
        assert props["search_query"] == "budget"
        assert props["results_count"] == 5

    async def test_capture_with_common_props_includes_session_id_when_provided(self):
        # Given: an enabled client
        with patch("src.analytics.client.posthog.Posthog") as MockPosthog:
            mock_ph = MagicMock()
            MockPosthog.return_value = mock_ph
            client = AnalyticsClient(api_key="phc_test_key_123")

            # When: capture_with_common_props is called with a session_id
            client.capture_with_common_props(
                distinct_id="user-xyz",
                event="search:query_submit",
                properties={"search_query": "budget"},
                is_internal=False,
                session_id="session-abc-123",
            )

        # Then: $session_id (PostHog's reserved key) is present on the event
        props = mock_ph.capture.call_args.kwargs["properties"]
        assert props["$session_id"] == "session-abc-123"

    async def test_capture_with_common_props_omits_session_id_when_none(self):
        # Given: an enabled client
        with patch("src.analytics.client.posthog.Posthog") as MockPosthog:
            mock_ph = MagicMock()
            MockPosthog.return_value = mock_ph
            client = AnalyticsClient(api_key="phc_test_key_123")

            # When: capture_with_common_props is called without a session_id
            client.capture_with_common_props(
                distinct_id="user-xyz",
                event="search:query_submit",
                properties={"search_query": "budget"},
                is_internal=False,
                session_id=None,
            )

        # Then: $session_id is absent (not present as null or empty)
        props = mock_ph.capture.call_args.kwargs["properties"]
        assert "$session_id" not in props

    async def test_capture_exception_is_swallowed(self):
        # Given: a client whose SDK raises on capture
        with patch("src.analytics.client.posthog.Posthog") as MockPosthog:
            mock_ph = MagicMock()
            mock_ph.capture.side_effect = RuntimeError("network error")
            MockPosthog.return_value = mock_ph
            client = AnalyticsClient(api_key="phc_test_key_123")

            # When / Then: exception does not propagate
            client._capture(
                distinct_id="user-abc",
                event="search:query_submit",
                properties={},
            )

    async def test_shutdown_calls_posthog_shutdown(self):
        # Given: an enabled client
        with patch("src.analytics.client.posthog.Posthog") as MockPosthog:
            mock_ph = MagicMock()
            MockPosthog.return_value = mock_ph
            client = AnalyticsClient(api_key="phc_test_key_123")

            # When: shutdown is called
            client.shutdown()

        # Then: the posthog client was shut down
        mock_ph.shutdown.assert_called_once()

    async def test_shutdown_exception_is_swallowed(self):
        # Given: a client whose SDK raises on shutdown
        with patch("src.analytics.client.posthog.Posthog") as MockPosthog:
            mock_ph = MagicMock()
            mock_ph.shutdown.side_effect = RuntimeError("flush error")
            MockPosthog.return_value = mock_ph
            client = AnalyticsClient(api_key="phc_test_key_123")

            # When / Then: shutdown does not propagate the exception
            client.shutdown()


class TestAnalyticsClientRequestHelpers:
    """get_distinct_id and is_internal_request extract from FastAPI Request."""

    def _make_request(self, headers: dict[str, str]) -> MagicMock:
        request = MagicMock()
        request.headers = headers
        return request

    async def test_get_distinct_id_returns_header_when_present(self):
        # Given: a client and a request with the PostHog distinct_id header
        client = AnalyticsClient(api_key="")
        request = self._make_request({"X-PostHog-Distinct-Id": "frontend-user-abc"})

        # When: get_distinct_id is called
        result = client.get_distinct_id(request)

        # Then: the header value is returned
        assert result == "frontend-user-abc"

    async def test_get_distinct_id_falls_back_to_uuid_when_header_absent(self):
        # Given: a request without the PostHog distinct_id header
        client = AnalyticsClient(api_key="")
        request = self._make_request({})

        # When: get_distinct_id is called
        result = client.get_distinct_id(request)

        # Then: a non-empty UUID string is returned
        assert result
        assert len(result) == 36  # standard UUID4 format

    async def test_get_distinct_id_returns_different_uuid_each_call(self):
        # Given: a request without the header (two calls)
        client = AnalyticsClient(api_key="")
        request = self._make_request({})

        # When: get_distinct_id is called twice
        id_1 = client.get_distinct_id(request)
        id_2 = client.get_distinct_id(request)

        # Then: each call produces a unique ID (not reused between requests)
        assert id_1 != id_2

    async def test_is_internal_request_true_when_header_is_true(self):
        # Given: a request with X-Internal-Request: true
        client = AnalyticsClient(api_key="")
        request = self._make_request({"X-Internal-Request": "true"})

        # When / Then
        assert client.is_internal_request(request) is True

    async def test_is_internal_request_false_when_header_absent(self):
        # Given: a request with no internal header
        client = AnalyticsClient(api_key="")
        request = self._make_request({})

        # When / Then
        assert client.is_internal_request(request) is False

    async def test_is_internal_request_false_when_header_value_is_not_true(self):
        # Given: header present but value is not "true"
        client = AnalyticsClient(api_key="")
        request = self._make_request({"X-Internal-Request": "yes"})

        # When / Then
        assert client.is_internal_request(request) is False

    async def test_get_session_id_returns_header_when_present(self):
        # Given: a request with the PostHog session_id header
        client = AnalyticsClient(api_key="")
        request = self._make_request({"X-PostHog-Session-Id": "session-xyz-789"})

        # When / Then
        assert client.get_session_id(request) == "session-xyz-789"

    async def test_get_session_id_returns_none_when_header_absent(self):
        # Given: a request without the PostHog session_id header
        client = AnalyticsClient(api_key="")
        request = self._make_request({})

        # When / Then: None (so callers can omit $session_id entirely)
        assert client.get_session_id(request) is None

    async def test_get_session_id_returns_none_when_header_empty(self):
        # Given: a request with an empty session_id header value
        client = AnalyticsClient(api_key="")
        request = self._make_request({"X-PostHog-Session-Id": ""})

        # When / Then
        assert client.get_session_id(request) is None

    async def test_get_navigation_source_returns_known_value(self):
        # Given: a request with X-Source set to a known navigation source
        client = AnalyticsClient(api_key="")
        request = self._make_request({"X-Source": "home"})

        # When / Then
        assert client.get_navigation_source(request) == "home"

    async def test_get_navigation_source_normalizes_case_and_whitespace(self):
        # Given: a request with mixed-case + whitespace around a known value
        client = AnalyticsClient(api_key="")
        request = self._make_request({"X-Source": "  SEARCH  "})

        # When / Then: normalized to lowercase trimmed value
        assert client.get_navigation_source(request) == "search"

    async def test_get_navigation_source_unknown_value_becomes_unknown(self):
        # Given: a request with X-Source set to a value not in the allowlist
        client = AnalyticsClient(api_key="")
        request = self._make_request({"X-Source": "weird-source"})

        # When / Then: defensively normalized to "unknown"
        assert client.get_navigation_source(request) == "unknown"

    async def test_get_navigation_source_absent_header_becomes_unknown(self):
        # Given: a request with no X-Source header
        client = AnalyticsClient(api_key="")
        request = self._make_request({})

        # When / Then
        assert client.get_navigation_source(request) == "unknown"
