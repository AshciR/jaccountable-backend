"""PostHog analytics client for jaacountable-backend."""

import os
from typing import Any
from uuid import uuid4

import posthog
from fastapi import Request
from loguru import logger


KNOWN_NAVIGATION_SOURCES: frozenset[str] = frozenset(
    {"home", "search", "related", "direct"}
)


class AnalyticsClient:
    """Wraps the PostHog Python SDK with graceful no-op behavior when unconfigured.

    If POSTHOG_API_KEY is absent or empty, all capture calls are silently
    dropped. This keeps analytics fully optional — no crashes, no side effects.

    The PostHog Python SDK sends events in background threads, so capture calls
    are safe to make from async route handlers without blocking the event loop.
    """

    def __init__(
        self,
        api_key: str | None = None,
        host: str | None = None,
        environment: str | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.getenv("POSTHOG_API_KEY", "")
        self._host = host or os.getenv("POSTHOG_HOST", "https://app.posthog.com")
        self.environment = environment or os.getenv("APP_ENV", "development")
        self._disabled = not bool(self._api_key)
        self._client: posthog.Posthog | None = None

        if self._disabled:
            logger.info("PostHog analytics disabled: POSTHOG_API_KEY not set")
        else:
            self._client = posthog.Posthog(
                project_api_key=self._api_key,
                host=self._host,
            )
            logger.info(
                "PostHog analytics enabled: host={}, env={}",
                self._host,
                self.environment,
            )

    @property
    def disabled(self) -> bool:
        return self._disabled

    def get_distinct_id(self, request: Request) -> str:
        """Extract the distinct_id for an event from the request.

        Prefers the X-PostHog-Distinct-Id header, which the frontend sets to
        its own PostHog distinct_id so backend events link to the same person.
        Falls back to a per-request UUID when the header is absent.

        Never uses client IP — shared NAT/VPN IPs would merge unrelated users
        into a single PostHog person profile.

        Update this method when user auth is added (return the user_id instead).
        """
        return request.headers.get("X-PostHog-Distinct-Id") or str(uuid4())

    def is_internal_request(self, request: Request) -> bool:
        """Return True if the request originated from internal infrastructure.

        Currently detected via the X-Internal-Request: true header. Callers
        (e.g. internal scripts, CI pipelines) set this header to flag that the
        traffic should not be counted as real user activity.

        Update this method to change the internal traffic detection strategy.
        """
        return request.headers.get("X-Internal-Request", "").lower() == "true"

    def get_navigation_source(self, request: Request) -> str:
        """Return the normalized navigation source for an incoming request.

        Reads the X-Source header (set by the frontend when linking to a page)
        and normalizes it against KNOWN_NAVIGATION_SOURCES. Unknown or missing
        values become "unknown" so the property is always present and stable
        for PostHog filtering.
        """
        raw = (request.headers.get("X-Source") or "").strip().lower()
        return raw if raw in KNOWN_NAVIGATION_SOURCES else "unknown"

    def capture_with_common_props(
        self,
        distinct_id: str,
        event: str,
        properties: dict[str, Any] | None = None,
        *,
        is_internal: bool = False,
    ) -> None:
        """Fire an analytics event with common properties automatically merged.

        Common properties injected into every event:
          - environment  (from APP_ENV env var, defaults to "development")
          - is_internal  (caller-provided, typically from is_internal_request())

        Use this method for all event tracking to ensure consistent properties.
        """
        common: dict[str, Any] = {
            "environment": self.environment,
            "is_internal": is_internal,
        }
        merged = {**common, **(properties or {})}
        self._capture(distinct_id=distinct_id, event=event, properties=merged)

    def shutdown(self) -> None:
        """Flush pending events and shut down background threads.

        Call this at application shutdown (lifespan teardown) to avoid losing
        events queued in PostHog's internal buffer. No-ops if disabled.
        """
        if self._disabled or self._client is None:
            return
        try:
            self._client.shutdown()
        except Exception:
            logger.exception("PostHog shutdown failed")

    def _capture(
        self,
        distinct_id: str,
        event: str,
        properties: dict[str, Any],
    ) -> None:
        if self._disabled or self._client is None:
            return
        try:
            self._client.capture(
                distinct_id=distinct_id,
                event=event,
                properties=properties,
            )
        except Exception:
            logger.exception("PostHog capture failed for event={}", event)


analytics_client = AnalyticsClient()
