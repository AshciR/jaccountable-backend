"""Utility functions for classification results."""
import asyncio
import random
import re
from collections.abc import Awaitable, Callable
from typing import TypeVar

from loguru import logger

from .models import ClassificationResult

T = TypeVar("T")

# Anthropic models wrap JSON output in ```json ... ``` markdown fences by
# default; OpenAI's response_format=json_object stripped these automatically.
_CODE_FENCE_PATTERN = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)


def strip_code_fence(text: str) -> str:
    """Strip a surrounding ```json ... ``` (or plain ``` ... ```) fence.

    Returns the input stripped of whitespace when no complete fence is present,
    so a malformed half-fence falls through to the downstream JSON parser with
    a useful error rather than silently producing a partial strip.
    """
    match = _CODE_FENCE_PATTERN.match(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def filter_relevant_classifications(
    results: list[ClassificationResult],
    min_confidence: float = 0.7
) -> list[ClassificationResult]:
    """
    Filter classification results to only relevant articles.

    An article is relevant if AT LEAST ONE classifier marks it as:
    - is_relevant = True
    - confidence >= min_confidence

    Args:
        results: Classification results from ClassificationService
        min_confidence: Minimum confidence threshold (default: 0.7)

    Returns:
        List of relevant classification results (may be empty)

    Example:
        >>> # Only store if relevant
        >>> relevant = filter_relevant_classifications(
        ...     results=classification_results,
        ...     min_confidence=0.7
        ... )
        >>> if relevant:
        ...     # Store article and classifications
        ...     pass
    """
    return [
        result
        for result in results
        if result.is_relevant and result.confidence >= min_confidence
    ]


async def retry_with_backoff(
    fn: Callable[[], Awaitable[T]],
    *,
    retry_on: tuple[type[Exception], ...],
    max_retries: int = 4,
    base_backoff: float = 2.0,
    label: str = "operation",
) -> T:
    """Execute an async callable with exponential backoff and jitter.

    Args:
        fn: Async callable to execute (no arguments; use a lambda to bind args).
        retry_on: Tuple of exception types that should trigger a retry.
        max_retries: Maximum number of attempts before re-raising.
        base_backoff: Base for exponential delay — attempt N waits base^N seconds
            (multiplied by random jitter between 0.5x and 1.5x).
        label: Human-readable name shown in log messages.

    Returns:
        The return value of *fn* on success.

    Raises:
        The last exception raised by *fn* if all attempts fail.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return await fn()
        except retry_on as e:
            if attempt < max_retries:
                backoff_time = (base_backoff ** attempt) * (0.5 + random.random())
                logger.warning(
                    "{} failed (attempt {}/{}) : {}. Retrying in {:.1f}s...",
                    label,
                    attempt,
                    max_retries,
                    e,
                    backoff_time,
                )
                await asyncio.sleep(backoff_time)
            else:
                logger.error(
                    "{} failed after {} attempts. Giving up.",
                    label,
                    max_retries,
                )
                raise
