"""Tests for ClassificationService."""
import asyncio
import time

import pytest
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    RateLimitError,
)

from src.article_classification.models import (
    ClassificationInput,
    ClassificationResult,
    ClassifierType,
)
from src.article_classification.services.classification_service import ClassificationService


class MockCorruptionClassifier:
    """Mock corruption classifier for testing parallel execution."""

    async def classify(self, article: ClassificationInput, max_text_chars: int | None = None) -> ClassificationResult:
        """Returns a mock corruption classification result."""
        return ClassificationResult(
            is_relevant=True,
            confidence=0.9,
            reasoning="OCG investigation",
            key_entities=["OCG"],
            classifier_type=ClassifierType.CORRUPTION,
            model_name="mock-corruption",
        )


class MockHurricaneClassifier:
    """Mock hurricane classifier for testing parallel execution."""

    async def classify(self, article: ClassificationInput, max_text_chars: int | None = None) -> ClassificationResult:
        """Returns a mock hurricane classification result."""
        return ClassificationResult(
            is_relevant=True,
            confidence=0.85,
            reasoning="Hurricane relief fund allocation",
            key_entities=["NEMA", "Ministry of Local Government"],
            classifier_type=ClassifierType.HURRICANE_RELIEF,
            model_name="mock-hurricane",
        )


class SlowClassifier:
    """Mock classifier with configurable delay for parallelism testing."""

    def __init__(self, classifier_type: ClassifierType, wait_time: float = 0.1):
        """
        Initialize slow classifier.

        Args:
            classifier_type: Type of classifier (CORRUPTION, HURRICANE_RELIEF, etc.)
            wait_time: Seconds to wait during classify() to simulate slow LLM call
        """
        self.classifier_type = classifier_type
        self.wait_time = wait_time

    async def classify(self, article: ClassificationInput, max_text_chars: int | None = None) -> ClassificationResult:
        """Simulate slow LLM call with configurable delay."""
        await asyncio.sleep(self.wait_time)
        return ClassificationResult(
            is_relevant=True,
            confidence=0.8,
            reasoning="Test",
            classifier_type=self.classifier_type,
            model_name="slow-model",
        )


class FailingClassifier:
    """Mock classifier that always raises an exception."""

    def __init__(self, error_message: str = "Classifier failed"):
        """
        Initialize failing classifier.

        Args:
            error_message: Custom error message for the exception
        """
        self.error_message = error_message

    async def classify(self, article: ClassificationInput, max_text_chars: int | None = None) -> ClassificationResult:
        """Always raises ValueError."""
        raise ValueError(self.error_message)


class FatalProviderErrorClassifier:
    """Mock classifier that raises a fatal litellm provider exception.

    Simulates the production failure mode where retry_with_backoff has exhausted
    its attempts and re-raised the underlying litellm error (auth / quota /
    connection). Service is expected to propagate these.
    """

    def __init__(self, exception: Exception):
        self.exception = exception

    async def classify(self, article: ClassificationInput, max_text_chars: int | None = None) -> ClassificationResult:
        raise self.exception


@pytest.fixture
def mock_corruption_classifier() -> MockCorruptionClassifier:
    """Mock corruption classifier."""
    return MockCorruptionClassifier()


@pytest.fixture
def mock_hurricane_classifier() -> MockHurricaneClassifier:
    """Mock hurricane classifier."""
    return MockHurricaneClassifier()


@pytest.fixture
def failing_classifier() -> FailingClassifier:
    """Mock classifier that raises exception."""
    return FailingClassifier()


@pytest.fixture
def rate_limit_classifier() -> FatalProviderErrorClassifier:
    """Mock classifier that raises litellm RateLimitError (e.g. quota exhausted)."""
    return FatalProviderErrorClassifier(
        RateLimitError(
            message="You exceeded your current quota",
            llm_provider="anthropic",
            model="claude-sonnet-4-6",
        )
    )


@pytest.fixture
def authentication_error_classifier() -> FatalProviderErrorClassifier:
    """Mock classifier that raises litellm AuthenticationError (e.g. bad API key)."""
    return FatalProviderErrorClassifier(
        AuthenticationError(
            message="Invalid API key",
            llm_provider="anthropic",
            model="claude-sonnet-4-6",
        )
    )


@pytest.fixture
def api_connection_error_classifier() -> FatalProviderErrorClassifier:
    """Mock classifier that raises litellm APIConnectionError (e.g. network outage)."""
    return FatalProviderErrorClassifier(
        APIConnectionError(
            message="Could not connect to provider",
            llm_provider="anthropic",
            model="claude-sonnet-4-6",
        )
    )


class TestClassificationServiceMultipleClassifiers:
    """Test service with multiple classifiers running in parallel."""

    async def test_runs_two_classifiers_in_parallel_returns_both_results(
        self,
        sample_corruption_article: ClassificationInput,
        mock_corruption_classifier: MockCorruptionClassifier,
        mock_hurricane_classifier: MockHurricaneClassifier,
    ):
        # Given: Service with corruption + hurricane classifiers
        service = ClassificationService(
            classifiers=[mock_corruption_classifier, mock_hurricane_classifier]
        )

        # When: Classifying an article
        results = await service.classify(sample_corruption_article)

        # Then: Returns 2 results with correct classifier types
        assert len(results) == 2

        corruption_result = next(
            r for r in results if r.classifier_type == ClassifierType.CORRUPTION
        )
        hurricane_result = next(
            r for r in results if r.classifier_type == ClassifierType.HURRICANE_RELIEF
        )

        assert corruption_result.is_relevant is True
        assert corruption_result.confidence == 0.9
        assert hurricane_result.is_relevant is True
        assert hurricane_result.confidence == 0.85

    async def test_runs_three_classifiers_in_parallel_returns_all_results(
        self,
        sample_corruption_article: ClassificationInput,
        mock_corruption_classifier: MockCorruptionClassifier,
        mock_hurricane_classifier: MockHurricaneClassifier,
    ):
        # Given: Service with 3 classifiers (create third classifier with slow wait time)
        third_classifier = SlowClassifier(
            classifier_type=ClassifierType.CORRUPTION, wait_time=0.0
        )

        service = ClassificationService(
            classifiers=[
                mock_corruption_classifier,
                mock_hurricane_classifier,
                third_classifier,
            ]
        )

        # When: Classifying an article
        results = await service.classify(sample_corruption_article)

        # Then: Returns 3 results
        assert len(results) == 3

    async def test_classifiers_run_in_parallel_not_sequentially(
        self, sample_corruption_article: ClassificationInput
    ):
        """Verify classifiers actually run in parallel, not sequentially."""
        # Given: Two classifiers that each take 0.1 seconds
        classifier1 = SlowClassifier(ClassifierType.CORRUPTION, wait_time=0.1)
        classifier2 = SlowClassifier(ClassifierType.HURRICANE_RELIEF, wait_time=0.1)

        service = ClassificationService(classifiers=[classifier1, classifier2])

        # When: Classifying article
        start = time.time()
        results = await service.classify(sample_corruption_article)
        elapsed = time.time() - start

        # Then: Total time is ~0.1s (parallel), not ~0.2s (sequential)
        assert len(results) == 2
        assert elapsed < 0.15  # Should be ~0.1s if parallel, ~0.2s if sequential


class TestClassificationServiceErrorHandling:
    """Test exception handling when classifiers fail."""

    async def test_one_classifier_fails_other_succeeds_returns_successful_result(
        self,
        sample_corruption_article: ClassificationInput,
        mock_corruption_classifier: MockCorruptionClassifier,
        failing_classifier: FailingClassifier,
    ):
        # Given: Service with one failing classifier and one working classifier
        service = ClassificationService(
            classifiers=[failing_classifier, mock_corruption_classifier]
        )

        # When: Classifying an article
        results = await service.classify(sample_corruption_article)

        # Then: Returns 1 result (from successful classifier), skips exception
        assert len(results) == 1
        assert results[0].classifier_type == ClassifierType.CORRUPTION
        assert results[0].is_relevant is True

    async def test_all_classifiers_fail_returns_empty_list(
        self,
        sample_corruption_article: ClassificationInput,
        failing_classifier: FailingClassifier,
    ):
        # Given: Service with two failing classifiers
        failing_classifier_2 = FailingClassifier("Second classifier failed")

        service = ClassificationService(
            classifiers=[failing_classifier, failing_classifier_2]
        )

        # When: Classifying an article
        results = await service.classify(sample_corruption_article)

        # Then: Returns empty list
        assert len(results) == 0


class TestClassificationServiceFatalProviderErrors:
    """Verify that fatal provider errors propagate so the batch job fails loud.

    These cover the regression that caused 4 days of silently-failing
    classification runs: when OpenAI returned quota-exhausted errors for every
    article, the service swallowed them and the workflow exited 0.
    """

    async def test_rate_limit_error_propagates(
        self,
        sample_corruption_article: ClassificationInput,
        rate_limit_classifier: FatalProviderErrorClassifier,
    ):
        # Given: A classifier whose retries have exhausted and raised RateLimitError
        service = ClassificationService(classifiers=[rate_limit_classifier])

        # When/Then: classify() re-raises so the caller can fail the batch
        with pytest.raises(RateLimitError):
            await service.classify(sample_corruption_article)

    async def test_authentication_error_propagates(
        self,
        sample_corruption_article: ClassificationInput,
        authentication_error_classifier: FatalProviderErrorClassifier,
    ):
        # Given: A classifier that raises AuthenticationError (bad / missing key)
        service = ClassificationService(classifiers=[authentication_error_classifier])

        # When/Then: classify() re-raises
        with pytest.raises(AuthenticationError):
            await service.classify(sample_corruption_article)

    async def test_api_connection_error_propagates(
        self,
        sample_corruption_article: ClassificationInput,
        api_connection_error_classifier: FatalProviderErrorClassifier,
    ):
        # Given: A classifier that raises APIConnectionError after retries
        service = ClassificationService(classifiers=[api_connection_error_classifier])

        # When/Then: classify() re-raises
        with pytest.raises(APIConnectionError):
            await service.classify(sample_corruption_article)

    async def test_fatal_error_propagates_even_when_other_classifier_succeeds(
        self,
        sample_corruption_article: ClassificationInput,
        mock_corruption_classifier: MockCorruptionClassifier,
        rate_limit_classifier: FatalProviderErrorClassifier,
    ):
        # Given: One classifier hits a quota error, another would have succeeded
        service = ClassificationService(
            classifiers=[rate_limit_classifier, mock_corruption_classifier]
        )

        # When/Then: The successful result is discarded — a quota-exhausted run
        # is a failed run, full stop. Better to fail loud than store a partial result.
        with pytest.raises(RateLimitError):
            await service.classify(sample_corruption_article)

    async def test_fatal_error_takes_precedence_over_non_fatal_error(
        self,
        sample_corruption_article: ClassificationInput,
        failing_classifier: FailingClassifier,
        rate_limit_classifier: FatalProviderErrorClassifier,
    ):
        # Given: One classifier raises a tolerable ValueError, another raises fatal
        service = ClassificationService(
            classifiers=[failing_classifier, rate_limit_classifier]
        )

        # When/Then: Fatal wins — non-fatal exceptions don't mask it
        with pytest.raises(RateLimitError):
            await service.classify(sample_corruption_article)

    async def test_non_fatal_value_error_still_swallowed(
        self,
        sample_corruption_article: ClassificationInput,
        mock_corruption_classifier: MockCorruptionClassifier,
        failing_classifier: FailingClassifier,
    ):
        # Given: A non-litellm exception (e.g. parser error on one article)
        service = ClassificationService(
            classifiers=[failing_classifier, mock_corruption_classifier]
        )

        # When: Classifying
        results = await service.classify(sample_corruption_article)

        # Then: Per-article failure is tolerated; successful classifier's result
        # is returned. Only provider-level outages halt the batch.
        assert len(results) == 1
        assert results[0].classifier_type == ClassifierType.CORRUPTION


class TestClassificationServiceEdgeCases:
    """Test edge cases and boundary conditions."""

    async def test_empty_classifier_list_returns_empty_results(
        self, sample_corruption_article: ClassificationInput
    ):
        # Given: Service with empty classifiers list
        service = ClassificationService(classifiers=[])

        # When: Classifying an article
        results = await service.classify(sample_corruption_article)

        # Then: Returns empty list immediately
        assert len(results) == 0

    async def test_single_classifier_returns_single_result(
        self,
        sample_corruption_article: ClassificationInput,
        mock_corruption_classifier: MockCorruptionClassifier,
    ):
        # Given: Service with only one classifier
        service = ClassificationService(classifiers=[mock_corruption_classifier])

        # When: Classifying an article
        results = await service.classify(sample_corruption_article)

        # Then: Returns 1 result
        assert len(results) == 1
        assert results[0].classifier_type == ClassifierType.CORRUPTION
