"""Base protocol for article classification strategies."""
import os
from typing import Protocol
from dotenv import load_dotenv
from .models import ClassificationInput, ClassificationResult, NormalizedEntity

# Load environment variables from .env file
load_dotenv()

APP_NAME = "jaccountable_backend"
CLASSIFICATION_MODEL = "anthropic/claude-haiku-4-5"
NORMALIZATION_MODEL = "anthropic/claude-haiku-4-5"
CLASSIFICATION_API_KEY = os.getenv("ANTHROPIC_CLASSIFICATION_API_KEY")

class ArticleClassifier(Protocol):
    """
    Protocol for article classification strategies using structural subtyping.

    This Protocol defines the interface for article classifiers without
    requiring inheritance. Any class that implements the classify() method
    with the correct signature can be used as an ArticleClassifier.

    This enables the Strategy Pattern with maximum flexibility:
    - No need to inherit from a base class
    - Duck typing with type safety
    - Easy to add new classifiers
    - Adapts Google ADK agents to a consistent interface

    Example:
        class MyClassifier:  # No inheritance needed!
            async def classify(self, article: ClassificationInput) -> ClassificationResult:
                # Implementation here
                pass

        # MyClassifier satisfies ArticleClassifier Protocol
    """

    async def classify(
        self, article: ClassificationInput, max_text_chars: int | None = None
    ) -> ClassificationResult:
        """
        Classify an article for relevance to a specific topic.

        Args:
            article: Article data with url, title, section, full_text, and
                    optional published_date.
            max_text_chars: If set, truncate full_text to this many characters
                    before building the LLM prompt. Reduces token usage.

        Returns:
            ClassificationResult with is_relevant (true/false), confidence,
            reasoning, key_entities, classifier_type, and model_name.

        Raises:
            ValueError: If article data is invalid or classification fails
        """
        ...


class EntityNormalizer(Protocol):
    """
    Protocol for entity normalization services using structural subtyping.

    This Protocol defines the interface for entity normalizers without requiring
    inheritance. Any class that implements the normalize() method with the correct
    signature can be used as an EntityNormalizer.

    Entity normalization converts raw entity names (with titles, variations, etc.)
    into canonical forms for consistency and deduplication across articles.

    Uses batch processing for efficiency (single LLM call for multiple entities).

    Example:
        class MyNormalizer:  # No inheritance needed!
            async def normalize(self, entities: list[str]) -> list[NormalizedEntity]:
                # Implementation here
                pass

        # MyNormalizer satisfies EntityNormalizer Protocol
    """

    async def normalize(
        self, entities: list[str]
    ) -> list[NormalizedEntity]:
        """
        Normalize a batch of entity names to canonical forms.

        Args:
            entities: List of original entity names to normalize.
                     Example: ["Hon. Ruel Reid", "OCG", "Ministry of Education"]

        Returns:
            List of NormalizedEntity objects with original/normalized pairs,
            per-entity confidence scores, and reasoning for each normalization.
            Example: [
                NormalizedEntity(
                    original_value="Hon. Ruel Reid",
                    normalized_value="ruel_reid",
                    confidence=0.95,
                    reason="Removed title 'Hon.' and standardized format"
                )
            ]

        Raises:
            ValueError: If entities list is empty or normalization fails
        """
        ...


class EntityCache(Protocol):
    """
    Protocol for entity normalization caches using structural subtyping.

    Cache Key: Normalized entity name (lowercase + whitespace collapsed)
    Cache Value: Complete NormalizedEntity object
    TTL: Configurable (default 14 days)
    Eviction: LRU (default 100k max entries)
    """

    async def get(self, entity_name: str) -> NormalizedEntity | None:
        """
        Retrieve normalized entity from cache.

        Args:
            entity_name: Original entity name (will be normalized for lookup)

        Returns:
            Cached NormalizedEntity or None if not found/expired
        """
        ...

    async def set(self, entity_name: str, normalized: NormalizedEntity) -> None:
        """
        Store normalized entity in cache with TTL.

        Args:
            entity_name: Original entity name (will be normalized for storage)
            normalized: NormalizedEntity to cache
        """
        ...

    async def get_many(self, entity_names: list[str]) -> dict[str, NormalizedEntity]:
        """
        Retrieve multiple entities (batch operation).

        Args:
            entity_names: List of original entity names

        Returns:
            Dict mapping original entity name → NormalizedEntity (hits only)
        """
        ...

    async def set_many(self, normalizations: dict[str, NormalizedEntity]) -> None:
        """
        Store multiple entities (batch operation) with TTL.

        Args:
            normalizations: Dict mapping original entity name → NormalizedEntity
        """
        ...

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dict with keys: hits, misses, size, max_size, hit_rate, ttl_seconds
        """
        ...
