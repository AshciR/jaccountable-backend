"""Tests for PipelineOrchestrationService."""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from src.orchestration.service import PipelineOrchestrationService
from src.article_extractor.models import ExtractedArticleContent
from src.article_extractor.base import ArticleExtractionService
from src.article_classification.models import ClassificationResult, ClassifierType
from src.article_classification.services.classification_service import ClassificationService
from src.article_persistence.service import PostgresArticlePersistenceService
from src.article_persistence.models.domain import ArticleStorageResult


class TestProcessArticleEdgeCases:
    """Unit tests for edge cases and error scenarios with mocks."""

    async def test_duplicate_url_not_stored_again(
        self,
        sample_extracted_content: ExtractedArticleContent,
    ):
        # Given: Mock services with duplicate article
        mock_extraction = Mock(spec=ArticleExtractionService)
        mock_extraction.extract_article_content = AsyncMock(
            return_value=sample_extracted_content
        )

        mock_classification = Mock(spec=ClassificationService)
        mock_classification.classify = AsyncMock(
            return_value=[
                ClassificationResult(
                    is_relevant=True,
                    confidence=0.9,
                    reasoning="Corruption investigation involving government contract",
                    classifier_type=ClassifierType.CORRUPTION,
                    model_name="gpt-4o-mini",
                    key_entities=["OCG", "Ministry of Education"],
                )
            ]
        )

        mock_persistence = Mock(spec=PostgresArticlePersistenceService)
        mock_persistence.store_article_with_classifications = AsyncMock(
            return_value=ArticleStorageResult(
                stored=False,  # Duplicate
                article_id=None,
                classification_count=0,
                article=None,
                classifications=[],
            )
        )

        service = PipelineOrchestrationService(
            extraction_service=mock_extraction,
            classification_service=mock_classification,
            persistence_service=mock_persistence,
        )

        mock_conn = Mock()
        url = "https://jamaica-gleaner.com/article/news/duplicate"
        section = "news"

        # When: Processing duplicate article
        result = await service.process_article(
            conn=mock_conn,
            url=url,
            section=section,
        )

        # Then: Not stored (duplicate)
        assert result.extracted is True
        assert result.classified is True
        assert result.relevant is True
        assert result.stored is False
        assert result.article_id is None
        assert result.classification_count == 0

    async def test_extraction_failure_returns_error_result(self):
        # Given: Mock services with extraction failure
        mock_extraction = Mock(spec=ArticleExtractionService)
        mock_extraction.extract_article_content = AsyncMock(
            side_effect=Exception("Failed to fetch URL: 404 Not Found")
        )

        service = PipelineOrchestrationService(
            extraction_service=mock_extraction,
        )

        mock_conn = Mock()
        url = "https://jamaica-gleaner.com/article/news/missing-404"
        section = "news"

        # When: Processing article with extraction failure
        result = await service.process_article(
            conn=mock_conn,
            url=url,
            section=section,
        )

        # Then: Error result returned, pipeline stopped
        assert result.extracted is False
        assert result.classified is False
        assert result.relevant is False
        assert result.stored is False
        assert result.article_id is None
        assert result.error is not None
        assert "Failed to extract article" in result.error

    async def test_classification_failure_returns_error_result(
        self,
        sample_extracted_content: ExtractedArticleContent,
    ):
        # Given: Mock services with classification failure
        mock_extraction = Mock(spec=ArticleExtractionService)
        mock_extraction.extract_article_content = AsyncMock(
            return_value=sample_extracted_content
        )

        mock_classification = Mock(spec=ClassificationService)
        mock_classification.classify = AsyncMock(
            side_effect=Exception("LLM API timeout")
        )

        service = PipelineOrchestrationService(
            extraction_service=mock_extraction,
            classification_service=mock_classification,
        )

        mock_conn = Mock()
        url = "https://jamaica-gleaner.com/article/news/test"
        section = "news"

        # When: Processing with classification failure
        result = await service.process_article(
            conn=mock_conn,
            url=url,
            section=section,
        )

        # Then: Error result with classified=False
        assert result.extracted is True
        assert result.classified is False
        assert result.relevant is False
        assert result.stored is False
        assert result.error is not None
        assert "Failed to classify article" in result.error

    async def test_storage_failure_returns_error_result(
        self,
        sample_extracted_content: ExtractedArticleContent,
    ):
        # Given: Mock services with storage failure
        mock_extraction = Mock(spec=ArticleExtractionService)
        mock_extraction.extract_article_content = AsyncMock(
            return_value=sample_extracted_content
        )

        mock_classification = Mock(spec=ClassificationService)
        mock_classification.classify = AsyncMock(
            return_value=[
                ClassificationResult(
                    is_relevant=True,
                    confidence=0.9,
                    reasoning="Corruption investigation",
                    classifier_type=ClassifierType.CORRUPTION,
                    model_name="gpt-4o-mini",
                    key_entities=["OCG"],
                )
            ]
        )

        mock_persistence = Mock(spec=PostgresArticlePersistenceService)
        mock_persistence.store_article_with_classifications = AsyncMock(
            side_effect=Exception("Database connection failed")
        )

        service = PipelineOrchestrationService(
            extraction_service=mock_extraction,
            classification_service=mock_classification,
            persistence_service=mock_persistence,
        )

        mock_conn = Mock()
        url = "https://jamaica-gleaner.com/article/news/test"
        section = "news"

        # When: Processing with storage failure
        result = await service.process_article(
            conn=mock_conn,
            url=url,
            section=section,
        )

        # Then: Error result with stored=False
        assert result.extracted is True
        assert result.classified is True
        assert result.relevant is True
        assert result.stored is False
        assert result.error is not None
        assert "Failed to store article" in result.error

    async def test_article_not_relevant_not_stored(
        self,
        sample_extracted_content: ExtractedArticleContent,
    ):
        # Given: Mock services with low-confidence classification
        mock_extraction = Mock(spec=ArticleExtractionService)
        mock_extraction.extract_article_content = AsyncMock(
            return_value=sample_extracted_content
        )

        mock_classification = Mock(spec=ClassificationService)
        mock_classification.classify = AsyncMock(
            return_value=[
                ClassificationResult(
                    is_relevant=False,  # Not relevant
                    confidence=0.3,
                    reasoning="Sports article about track championship",
                    classifier_type=ClassifierType.CORRUPTION,
                    model_name="gpt-4o-mini",
                    key_entities=[],
                )
            ]
        )

        service = PipelineOrchestrationService(
            extraction_service=mock_extraction,
            classification_service=mock_classification,
        )

        mock_conn = Mock()
        url = "https://jamaica-gleaner.com/article/sports/test"
        section = "sports"

        # When: Processing non-relevant article
        result = await service.process_article(
            conn=mock_conn,
            url=url,
            section=section,
        )

        # Then: Not stored, no error
        assert result.extracted is True
        assert result.classified is True
        assert result.relevant is False
        assert result.stored is False
        assert result.article_id is None
        assert result.classification_count == 0
        assert result.error is None

    async def test_relevant_article_stored_via_lazy_connection(
        self,
        sample_extracted_content: ExtractedArticleContent,
    ):
        # Given: Mock services that produce a relevant result
        mock_extraction = Mock(spec=ArticleExtractionService)
        mock_extraction.extract_article_content = AsyncMock(
            return_value=sample_extracted_content
        )

        mock_classification = Mock(spec=ClassificationService)
        mock_classification.classify = AsyncMock(
            return_value=[
                ClassificationResult(
                    is_relevant=True,
                    confidence=0.9,
                    reasoning="Corruption investigation involving government contract",
                    classifier_type=ClassifierType.CORRUPTION,
                    model_name="gpt-4o-mini",
                    key_entities=["OCG"],
                )
            ]
        )

        mock_persistence = Mock(spec=PostgresArticlePersistenceService)
        mock_persistence.store_article_with_classifications = AsyncMock(
            return_value=ArticleStorageResult(
                stored=True,
                article_id=42,
                classification_count=1,
                article=None,
                classifications=[],
            )
        )

        service = PipelineOrchestrationService(
            extraction_service=mock_extraction,
            classification_service=mock_classification,
            persistence_service=mock_persistence,
        )

        # Patch db_config.connection() so no real pool is needed
        mock_conn = MagicMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        @asynccontextmanager
        async def mock_connection():
            yield mock_conn

        url = "https://jamaica-gleaner.com/article/news/test-lazy"
        section = "news"

        # When: process_article called WITHOUT conn (triggers lazy acquisition)
        with patch("src.orchestration.service.db_config.connection", mock_connection):
            result = await service.process_article(url=url, section=section)

        # Then: Article stored via lazily-acquired connection
        assert result.extracted is True
        assert result.classified is True
        assert result.relevant is True
        assert result.stored is True
        assert result.article_id == 42
        assert result.error is None

        # And: persistence was called with the lazily-acquired connection
        mock_persistence.store_article_with_classifications.assert_called_once()
        call_kwargs = mock_persistence.store_article_with_classifications.call_args.kwargs
        assert call_kwargs["conn"] is mock_conn


class TestProcessArticleFallbackPublishedDate:
    """Discovery date is used only when the extractor finds no published date."""

    @staticmethod
    def _build_service(extracted: ExtractedArticleContent) -> tuple[PipelineOrchestrationService, Mock]:
        mock_extraction = Mock(spec=ArticleExtractionService)
        mock_extraction.extract_article_content = AsyncMock(return_value=extracted)

        mock_classification = Mock(spec=ClassificationService)
        mock_classification.classify = AsyncMock(
            return_value=[
                ClassificationResult(
                    is_relevant=True,
                    confidence=0.9,
                    reasoning="Corruption investigation involving government contract",
                    classifier_type=ClassifierType.CORRUPTION,
                    model_name="gpt-4o-mini",
                    key_entities=["OCG"],
                )
            ]
        )

        mock_persistence = Mock(spec=PostgresArticlePersistenceService)
        mock_persistence.store_article_with_classifications = AsyncMock(
            return_value=ArticleStorageResult(
                stored=True,
                article_id=42,
                classification_count=1,
                article=None,
                classifications=[],
            )
        )

        service = PipelineOrchestrationService(
            extraction_service=mock_extraction,
            classification_service=mock_classification,
            persistence_service=mock_persistence,
        )
        return service, mock_persistence

    async def test_fallback_used_when_extractor_finds_no_date(
        self,
        sample_extracted_content: ExtractedArticleContent,
    ):
        # Given: extractor returns content with no published date
        extracted = sample_extracted_content.model_copy(update={"published_date": None})
        service, mock_persistence = self._build_service(extracted)
        fallback = datetime(2026, 9, 25, tzinfo=timezone.utc)

        # When: processing with a fallback date from discovery
        await service.process_article(
            conn=MagicMock(),
            url="https://jamaica-gleaner.com/article/news/20260925/test",
            section="news",
            fallback_published_date=fallback,
        )

        # Then: the stored content carries the fallback date
        call_kwargs = mock_persistence.store_article_with_classifications.call_args.kwargs
        assert call_kwargs["extracted"].published_date == fallback

    async def test_extracted_date_wins_over_fallback(
        self,
        sample_extracted_content: ExtractedArticleContent,
    ):
        # Given: extractor returns content with a published date
        service, mock_persistence = self._build_service(sample_extracted_content)
        fallback = datetime(2026, 9, 25, tzinfo=timezone.utc)

        # When: processing with a different fallback date
        await service.process_article(
            conn=MagicMock(),
            url="https://jamaica-gleaner.com/article/news/20251201/test",
            section="news",
            fallback_published_date=fallback,
        )

        # Then: the extracted date is kept
        call_kwargs = mock_persistence.store_article_with_classifications.call_args.kwargs
        assert call_kwargs["extracted"].published_date == sample_extracted_content.published_date
