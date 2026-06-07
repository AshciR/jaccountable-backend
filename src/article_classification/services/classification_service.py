"""Classification service that orchestrates article classification."""
import asyncio
from loguru import logger
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    RateLimitError,
)

from src.article_classification.base import ArticleClassifier
from src.article_classification.models import ClassificationInput, ClassificationResult

# Provider-level failures that mean the entire run cannot succeed (exhausted
# credits, bad API key, network outage that survived all retries). These must
# propagate so the batch script exits non-zero instead of silently classifying
# every article as "no result".
FATAL_PROVIDER_ERRORS = (AuthenticationError, RateLimitError, APIConnectionError)


class ClassificationService:
    """
    Service that orchestrates article classification using multiple classifiers.

    This service runs all classifiers in parallel and returns all results,
    enabling multi-label classification. For example, an article about
    "government misuses hurricane relief funds" will be classified by both
    the corruption classifier AND the hurricane relief classifier.

    Usage:
        # With multiple classifiers
        from src.article_classification.agents import CorruptionClassifierAdapter

        corruption_classifier = CorruptionClassifierAdapter()
        # hurricane_classifier = HurricaneReliefClassifierAdapter()  # Future

        service = ClassificationService(classifiers=[corruption_classifier])

        # Classify article
        article = ClassificationInput(
            url="https://example.com/article",
            title="OCG Probes Hurricane Relief Fund Misuse",
            section="news",
            full_text="The Office of the Contractor General has launched..."
        )
        results = await service.classify(article)

        # Store relevant classifications
        for result in results:
            if result.is_relevant and result.confidence >= 0.7:
                # Store classification in database
                pass
    """

    def __init__(self, classifiers: list[ArticleClassifier]):
        """
        Initialize classification service with multiple classifiers.

        Args:
            classifiers: List of classifier instances implementing ArticleClassifier Protocol
        """
        self.classifiers = classifiers

    async def classify(
        self, article: ClassificationInput, max_text_chars: int | None = None
    ) -> list[ClassificationResult]:
        """
        Classify article using all classifiers in parallel.

        Runs all classifiers concurrently to enable multi-label classification.
        Each classifier independently determines relevance.

        Args:
            article: Article data with url, title, section, full_text, etc.
            max_text_chars: If set, truncate full_text to this many characters
                    before building the LLM prompt. Reduces token usage.

        Returns:
            List of ClassificationResults from all classifiers. Each result
            includes is_relevant (true/false), confidence, reasoning, etc.
            Empty list if no classifiers configured.

        Raises:
            ValueError: If article data is invalid
        """
        if not self.classifiers:
            return []

        # Run all classifiers in parallel
        results = await asyncio.gather(
            *[classifier.classify(article, max_text_chars=max_text_chars) for classifier in self.classifiers],
            return_exceptions=True,
        )

        classified_results = []
        for i, result in enumerate(results):
            classifier_name = self.classifiers[i].__class__.__name__

            if isinstance(result, FATAL_PROVIDER_ERRORS):
                logger.error(
                    f"Classifier {classifier_name} hit fatal provider error for article {article.url}: "
                    f"{type(result).__name__}: {result}"
                )
                raise result

            if isinstance(result, Exception):
                logger.warning(
                    f"Classifier {classifier_name} failed for article {article.url}: {type(result).__name__}: {result}"
                )
                continue

            classified_results.append(result)

        return classified_results
