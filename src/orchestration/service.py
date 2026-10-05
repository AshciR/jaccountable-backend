"""
Orchestration service for article processing pipeline.

Coordinates the full workflow: Extract → Classify → Filter → Store
"""
import time
from datetime import datetime

import asyncpg
from loguru import logger

from config.database import db_config

from src.article_extractor.models import ExtractedArticleContent
from src.article_extractor.base import ArticleExtractionService
from src.article_extractor.service import DefaultArticleExtractionService
from src.article_classification.models import ClassificationInput, ClassificationResult, NormalizedEntity
from src.article_classification.services.classification_service import ClassificationService
from src.article_classification.classifiers.corruption_classifier import CorruptionClassifier
from src.article_classification.services.entity_normalizer_service import EntityNormalizerService
from src.article_classification.services.in_memory_entity_cache import get_entity_cache
from src.article_classification.converters import extracted_content_to_classification_input
from src.article_classification.utils import filter_relevant_classifications
from src.article_persistence.service import PostgresArticlePersistenceService
from .models import OrchestrationResult


class PipelineOrchestrationService:
    """
    Orchestrates the full article processing pipeline.

    This service coordinates the workflow:
    1. Extract article content from URL
    2. Convert to classification input
    3. Classify article with multiple classifiers
    4. Filter relevant classifications
    5. Store article and classifications if relevant

    The service uses dependency injection for testability and flexibility.
    All dependencies default to production implementations if not provided.

    Example:
        # Default configuration (production)
        service = PipelineOrchestrationService()
        async with db_config.connection() as conn:
            result = await service.process_article(
                conn=conn,
                url="https://jamaica-gleaner.com/article/news/...",
                section="news",
            )

        # Custom configuration (testing)
        service = PipelineOrchestrationService(
            extraction_service=MockExtractionService(),
            classification_service=MockClassificationService(),
            persistence_service=MockPersistenceService(),
        )
    """

    def __init__(
        self,
        extraction_service: ArticleExtractionService | None = None,
        classification_service: ClassificationService | None = None,
        persistence_service: PostgresArticlePersistenceService | None = None,
        entity_normalizer: EntityNormalizerService | None = None,
    ):
        """
        Initialize orchestration service with dependencies.

        Args:
            extraction_service: Service for extracting article content
                (default: DefaultArticleExtractionService())
            classification_service: Service for classifying articles
                (default: ClassificationService with CorruptionClassifier)
            persistence_service: Service for storing articles and classifications
                (default: PostgresArticlePersistenceService())
            entity_normalizer: Service for normalizing entity names
                (default: EntityNormalizerService())
        """
        self.extraction_service = extraction_service or DefaultArticleExtractionService()
        self.classification_service = classification_service or ClassificationService(
            classifiers=[
                CorruptionClassifier(),
            ]
        )
        self.persistence_service = (
            persistence_service or PostgresArticlePersistenceService()
        )
        # Inject singleton cache into entity normalizer (production config)
        self.entity_normalizer = entity_normalizer or EntityNormalizerService(
            cache=get_entity_cache()  # Singleton cache shared across all instances
        )

    async def __aenter__(self):
        """
        Initialize extraction service with HTTP connection pooling.

        If the extraction service supports async context manager (has __aenter__),
        this method initializes it for connection pooling during batch processing.

        Returns:
            Self to enable async context manager pattern
        """
        if hasattr(self.extraction_service, "__aenter__"):
            await self.extraction_service.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Clean up extraction service resources.

        If the extraction service supports async context manager (has __aexit__),
        this method closes HTTP connections and cleans up resources.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred
        """
        if hasattr(self.extraction_service, "__aexit__"):
            await self.extraction_service.__aexit__(exc_type, exc_val, exc_tb)

    async def process_article(
        self,
        url: str,
        section: str,
        news_source_id: int = 1,
        min_confidence: float = 0.7,
        max_text_chars: int = 4000,
        conn: asyncpg.Connection | None = None,
        fallback_published_date: datetime | None = None,
    ) -> OrchestrationResult:
        """
        Process an article through the full pipeline: Extract → Classify → Store.

        Emits ONE canonical log line at the end with complete telemetry.

        Args:
            url: Article URL to process
            section: Article section/category (e.g., "news", "lead-stories")
            news_source_id: Database ID of news source (default: 1 for Jamaica Gleaner)
            min_confidence: Minimum confidence threshold for relevance (default: 0.7)
            max_text_chars: Truncate article full_text to this many characters
                before classification. Reduces token usage to stay within
                rate limits. (default: 4000)
            conn: Optional database connection. If None, a connection is acquired
                  lazily inside _step_store and released immediately after storing.
                  Pass explicitly to control the transaction lifecycle (e.g. dry-run
                  rollback or when the caller manages connection scope).
            fallback_published_date: Date to use when the extractor finds no
                  published date on the page (e.g. the date from discovery via
                  RSS/sitemap/URL). Ignored when the extractor finds one.

        Returns:
            OrchestrationResult with processing outcome and metadata

        Example:
            >>> service = PipelineOrchestrationService()
            >>> # Lazy connection (recommended for batch processing)
            >>> result = await service.process_article(
            ...     url="https://jamaica-gleaner.com/article/news/...",
            ...     section="news",
            ... )
            >>> # Explicit connection (for dry-run / caller-managed transactions)
            >>> async with db_config.connection() as conn:
            ...     result = await service.process_article(
            ...         conn=conn,
            ...         url="https://jamaica-gleaner.com/article/news/...",
            ...         section="news",
            ...     )
        """
        telemetry: dict[str, str | int | float] = {
            "url": url,
            "section": section,
            "news_source_id": news_source_id,
            "min_confidence": min_confidence,
        }
        pipeline_start = time.perf_counter()

        try:
            # Step 1: Extract article content
            success, result = await self._step_extract(
                extraction_service=self.extraction_service,
                url=url,
                section=section,
                telemetry=telemetry,
                pipeline_start=pipeline_start,
            )
            if not success:
                return result  # type: ignore
            extracted: ExtractedArticleContent = result  # type: ignore

            if extracted.published_date is None and fallback_published_date is not None:
                extracted = extracted.model_copy(
                    update={"published_date": fallback_published_date}
                )
                telemetry["published_date_source"] = "fallback"
            elif extracted.published_date is not None:
                telemetry["published_date_source"] = "extracted"
            else:
                telemetry["published_date_source"] = "missing"

            # Step 2: Convert to classification input
            success, result = self._step_convert(
                extracted=extracted,
                url=url,
                section=section,
                telemetry=telemetry,
                pipeline_start=pipeline_start,
            )
            if not success:
                return result  # type: ignore
            classification_input: ClassificationInput = result  # type: ignore

            # Step 3: Classify article
            success, result = await self._step_classify(
                classification_service=self.classification_service,
                classification_input=classification_input,
                url=url,
                section=section,
                max_text_chars=max_text_chars,
                telemetry=telemetry,
                pipeline_start=pipeline_start,
            )
            if not success:
                return result  # type: ignore
            classification_results: list[ClassificationResult] = result  # type: ignore

            # Step 4: Filter relevant classifications
            success, result = self._step_filter(
                classification_results=classification_results,
                min_confidence=min_confidence,
                url=url,
                section=section,
                telemetry=telemetry,
                pipeline_start=pipeline_start,
            )
            if not success:
                return result  # type: ignore
            relevant_results: list[ClassificationResult] = result  # type: ignore

            # Step 4.5: Normalize entities from relevant classifications
            normalized_entities = await self._step_normalize_entities(
                entity_normalizer=self.entity_normalizer,
                relevant_results=relevant_results,
                url=url,
                section=section,
                telemetry=telemetry,
            )

            # Step 5: Store article and classifications
            return await self._step_store(
                persistence_service=self.persistence_service,
                conn=conn,
                extracted=extracted,
                url=url,
                section=section,
                relevant_results=relevant_results,
                classification_results=classification_results,
                normalized_entities=normalized_entities,
                news_source_id=news_source_id,
                telemetry=telemetry,
                pipeline_start=pipeline_start,
            )

        except Exception as e:
            # Unexpected error - emit canonical log with whatever we collected
            telemetry.update({
                "error": str(e),
                "error_type": type(e).__name__,
                "error_stage": "unexpected",
                "total_duration_ms": round((time.perf_counter() - pipeline_start) * 1000, 2),
            })
            logger.bind(**telemetry).error("canonical-log-line", exc_info=True)
            raise

    async def _step_extract(
        self,
        extraction_service: ArticleExtractionService,
        url: str,
        section: str,
        telemetry: dict,
        pipeline_start: float,
    ) -> tuple[bool, ExtractedArticleContent | OrchestrationResult]:
        extraction_start = time.perf_counter()
        try:
            extracted = await extraction_service.extract_article_content(url)
        except Exception as e:
            telemetry.update({
                "extraction_duration_ms": round((time.perf_counter() - extraction_start) * 1000, 2),
                "extracted": False,
                "classified": False,
                "relevant": False,
                "stored": False,
                "error": f"Failed to extract article: {e}",
                "error_stage": "extraction",
                "total_duration_ms": round((time.perf_counter() - pipeline_start) * 1000, 2),
            })
            logger.bind(**telemetry).error("canonical-log-line")
            return (
                False,
                OrchestrationResult(
                    url=url,
                    section=section,
                    extracted=False,
                    classified=False,
                    relevant=False,
                    stored=False,
                    article_id=None,
                    classification_count=0,
                    classification_results=[],
                    error=f"Failed to extract article: {e}",
                ),
            )

        telemetry["extraction_duration_ms"] = round((time.perf_counter() - extraction_start) * 1000, 2)
        telemetry.update({
            "extracted": True,
            "extracted_title": extracted.title[:100],
        })
        return (True, extracted)

    def _step_convert(
        self,
        extracted: ExtractedArticleContent,
        url: str,
        section: str,
        telemetry: dict,
        pipeline_start: float,
    ) -> tuple[bool, ClassificationInput | OrchestrationResult]:
        try:
            classification_input = extracted_content_to_classification_input(
                extracted=extracted,
                url=url,
                section=section,
            )
        except Exception as e:
            telemetry.update({
                "classified": False,
                "relevant": False,
                "stored": False,
                "error": f"Failed to convert to classification input: {e}",
                "error_stage": "conversion",
                "total_duration_ms": round((time.perf_counter() - pipeline_start) * 1000, 2),
            })
            logger.bind(**telemetry).error("canonical-log-line")
            return (
                False,
                OrchestrationResult(
                    url=url,
                    section=section,
                    extracted=True,
                    classified=False,
                    relevant=False,
                    stored=False,
                    article_id=None,
                    classification_count=0,
                    classification_results=[],
                    error=f"Failed to convert to classification input: {e}",
                ),
            )

        return (True, classification_input)

    async def _step_classify(
        self,
        classification_service: ClassificationService,
        classification_input: ClassificationInput,
        url: str,
        section: str,
        max_text_chars: int,
        telemetry: dict,
        pipeline_start: float,
    ) -> tuple[bool, list[ClassificationResult] | OrchestrationResult]:
        classification_start = time.perf_counter()
        try:
            classification_results = await classification_service.classify(
                classification_input, max_text_chars=max_text_chars
            )
        except Exception as e:
            telemetry.update({
                "classification_duration_ms": round((time.perf_counter() - classification_start) * 1000, 2),
                "classified": False,
                "relevant": False,
                "stored": False,
                "error": f"Failed to classify article: {e}",
                "error_stage": "classification",
                "total_duration_ms": round((time.perf_counter() - pipeline_start) * 1000, 2),
            })
            logger.bind(**telemetry).error("canonical-log-line")
            return (
                False,
                OrchestrationResult(
                    url=url,
                    section=section,
                    extracted=True,
                    classified=False,
                    relevant=False,
                    stored=False,
                    article_id=None,
                    classification_count=0,
                    classification_results=[],
                    error=f"Failed to classify article: {e}",
                ),
            )

        telemetry["classification_duration_ms"] = round((time.perf_counter() - classification_start) * 1000, 2)
        telemetry.update({
            "classified": True,
            "classifier_count": len(classification_results),
        })
        for cls_result in classification_results:
            prefix = cls_result.classifier_type.value.lower()
            telemetry[f"{prefix}_relevant"] = cls_result.is_relevant
            telemetry[f"{prefix}_confidence"] = cls_result.confidence
            telemetry[f"{prefix}_model"] = cls_result.model_name

        return (True, classification_results)

    def _step_filter(
        self,
        classification_results: list[ClassificationResult],
        min_confidence: float,
        url: str,
        section: str,
        telemetry: dict,
        pipeline_start: float,
    ) -> tuple[bool, list[ClassificationResult] | OrchestrationResult]:
        try:
            relevant_results = filter_relevant_classifications(
                results=classification_results,
                min_confidence=min_confidence,
            )
        except Exception as e:
            telemetry.update({
                "relevant": False,
                "stored": False,
                "relevant_classifiers": 0,
                "error": f"Failed to filter classifications: {e}",
                "error_stage": "filtering",
                "total_duration_ms": round((time.perf_counter() - pipeline_start) * 1000, 2),
            })
            logger.bind(**telemetry).error("canonical-log-line")
            return (
                False,
                OrchestrationResult(
                    url=url,
                    section=section,
                    extracted=True,
                    classified=True,
                    relevant=False,
                    stored=False,
                    article_id=None,
                    classification_count=0,
                    classification_results=classification_results,
                    error=f"Failed to filter classifications: {e}",
                ),
            )

        if not relevant_results:
            telemetry.update({
                "relevant": False,
                "stored": False,
                "relevant_classifiers": 0,
                "total_duration_ms": round((time.perf_counter() - pipeline_start) * 1000, 2),
            })
            logger.bind(**telemetry).info("canonical-log-line")
            return (
                False,
                OrchestrationResult(
                    url=url,
                    section=section,
                    extracted=True,
                    classified=True,
                    relevant=False,
                    stored=False,
                    article_id=None,
                    classification_count=0,
                    classification_results=classification_results,
                    error=None,
                ),
            )

        telemetry.update({
            "relevant": True,
            "relevant_classifiers": len(relevant_results),
        })
        return (True, relevant_results)

    async def _step_normalize_entities(
        self,
        entity_normalizer: EntityNormalizerService,
        relevant_results: list[ClassificationResult],
        url: str,
        section: str,
        telemetry: dict,
    ) -> list[NormalizedEntity]:
        entity_normalization_start = time.perf_counter()
        try:
            unique_entities: set[str] = set()
            for classification in relevant_results:
                unique_entities.update(classification.key_entities)

            if not unique_entities:
                telemetry["entity_count"] = 0
                telemetry["entity_normalization_duration_ms"] = round((time.perf_counter() - entity_normalization_start) * 1000, 2)
                return []

            normalized: list[NormalizedEntity] = await entity_normalizer.normalize(list(unique_entities))
            telemetry["entity_count"] = len(normalized)
        except Exception as e:
            logger.bind(url=url, section=section, error_type=type(e).__name__).warning(
                f"Entity normalization failed - continuing: {e}"
            )
            normalized = []
            telemetry["entity_count"] = 0
            telemetry["entity_normalization_error"] = str(e)

        telemetry["entity_normalization_duration_ms"] = round((time.perf_counter() - entity_normalization_start) * 1000, 2)
        return normalized

    async def _step_store(
        self,
        persistence_service: PostgresArticlePersistenceService,
        conn: asyncpg.Connection | None,
        extracted: ExtractedArticleContent,
        url: str,
        section: str,
        relevant_results: list[ClassificationResult],
        classification_results: list[ClassificationResult],
        normalized_entities: list[NormalizedEntity],
        news_source_id: int,
        telemetry: dict,
        pipeline_start: float,
    ) -> OrchestrationResult:
        storage_start = time.perf_counter()
        if conn is not None:
            # Caller-managed connection (tests, dry-run, validate scripts)
            return await self._do_store(
                persistence_service=persistence_service,
                conn=conn,
                extracted=extracted,
                url=url,
                section=section,
                relevant_results=relevant_results,
                classification_results=classification_results,
                normalized_entities=normalized_entities,
                news_source_id=news_source_id,
                telemetry=telemetry,
                pipeline_start=pipeline_start,
                storage_start=storage_start,
            )
        else:
            # Lazy acquisition: connection held only for the duration of the store
            async with db_config.connection() as acquired_conn:
                return await self._do_store(
                    persistence_service=persistence_service,
                    conn=acquired_conn,
                    extracted=extracted,
                    url=url,
                    section=section,
                    relevant_results=relevant_results,
                    classification_results=classification_results,
                    normalized_entities=normalized_entities,
                    news_source_id=news_source_id,
                    telemetry=telemetry,
                    pipeline_start=pipeline_start,
                    storage_start=storage_start,
                )

    async def _do_store(
        self,
        persistence_service: PostgresArticlePersistenceService,
        conn: asyncpg.Connection,
        extracted: ExtractedArticleContent,
        url: str,
        section: str,
        relevant_results: list[ClassificationResult],
        classification_results: list[ClassificationResult],
        normalized_entities: list[NormalizedEntity],
        news_source_id: int,
        telemetry: dict,
        pipeline_start: float,
        storage_start: float,
    ) -> OrchestrationResult:
        """Execute the store against a resolved (non-None) connection."""
        try:
            storage_result = await persistence_service.store_article_with_classifications(
                conn=conn,
                extracted=extracted,
                url=url,
                section=section,
                relevant_classifications=relevant_results,
                normalized_entities=normalized_entities,
                news_source_id=news_source_id,
            )
        except Exception as e:
            error_msg = f"Failed to store article: {e}"
            telemetry.update({
                "storage_duration_ms": round((time.perf_counter() - storage_start) * 1000, 2),
                "stored": False,
                "article_id": None,
                "classification_count": 0,
                "error": error_msg,
                "error_stage": "storage",
                "total_duration_ms": round((time.perf_counter() - pipeline_start) * 1000, 2),
            })
            logger.bind(**telemetry).error("canonical-log-line")
            return OrchestrationResult(
                url=url,
                section=section,
                extracted=True,
                classified=True,
                relevant=True,
                stored=False,
                article_id=None,
                classification_count=0,
                classification_results=classification_results,
                error=error_msg,
            )

        telemetry["storage_duration_ms"] = round((time.perf_counter() - storage_start) * 1000, 2)
        telemetry.update({
            "stored": storage_result.stored,
            "article_id": storage_result.article_id,
            "classification_count": storage_result.classification_count,
        })
        telemetry["total_duration_ms"] = round((time.perf_counter() - pipeline_start) * 1000, 2)

        if storage_result.stored:
            logger.bind(**telemetry).info("canonical-log-line")
            return OrchestrationResult(
                url=url,
                section=section,
                extracted=True,
                classified=True,
                relevant=True,
                stored=True,
                article_id=storage_result.article_id,
                classification_count=storage_result.classification_count,
                classification_results=classification_results,
                error=None,
            )
        else:
            # Duplicate article case
            logger.bind(**telemetry).warning("canonical-log-line")
            return OrchestrationResult(
                url=url,
                section=section,
                extracted=True,
                classified=True,
                relevant=True,
                stored=False,
                article_id=None,
                classification_count=0,
                classification_results=classification_results,
                error=None,
            )
