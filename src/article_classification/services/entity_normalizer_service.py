"""Service for normalizing entity names using the normalization agent."""
import json
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session, BaseSessionService
from google.genai import types
from google.genai.types import Content
from loguru import logger

from src.article_classification.agents.normalization_agent import normalization_agent
from src.article_classification.models import NormalizedEntity
from src.article_classification.base import APP_NAME, EntityCache
from src.article_classification.utils import strip_code_fence


class EntityNormalizerService:
    """Wraps normalization_agent and implements EntityNormalizer protocol."""

    agent: LlmAgent
    session_service: BaseSessionService
    runner: Runner
    cache: EntityCache | None

    def __init__(
        self,
        agent: LlmAgent | None = None,
        session_service: BaseSessionService | None = None,
        runner: Runner | None = None,
        cache: EntityCache | None = None,
    ):
        """
        Initialize the normalizer service.

        Args:
            agent: LLM agent (defaults to normalization_agent)
            session_service: Session service (defaults to InMemorySessionService)
            runner: Pre-configured runner (defaults to new Runner with agent+service)
            cache: Entity cache for reducing LLM calls (optional, defaults to None)
        """
        self.agent = agent or normalization_agent
        self.session_service = session_service or InMemorySessionService()
        self.runner = runner or Runner(
            app_name=APP_NAME,
            agent=self.agent,
            session_service=self.session_service
        )
        self.cache = cache
        logger.info(f"Initialized EntityNormalizerService (cache: {'enabled' if cache else 'disabled'})")

    async def normalize(self, entities: list[str]) -> list[NormalizedEntity]:
        """Normalize a batch of entities using cache + normalization agent."""
        if not entities:
            raise ValueError("entities list cannot be empty")

        # Step 1: Check cache
        cached_results: dict[str, NormalizedEntity]
        uncached_entities: list[str]
        cached_results, uncached_entities = await self._get_cached_entities(entities, self.cache)

        # Step 2: If all cached, return early
        if not uncached_entities:
            logger.info("All entities found in cache (no LLM call needed)")
            await self._log_cache_stats(self.cache)
            return [cached_results[e] for e in entities]

        # Step 3: Normalize uncached entities via LLM
        logger.info(f"Calling LLM to normalize {len(uncached_entities)} entities")

        # Create new session for this normalization
        session: Session = await self.session_service.create_session(
            app_name=APP_NAME,
            user_id="entity_normalizer"
        )

        # Build prompt for normalization agent
        entities_str = ", ".join(f'"{e}"' for e in uncached_entities)
        prompt = f"Normalize these entities: {entities_str}"

        # Call normalization agent using runner
        response = await self._call_agent_async(prompt, session.user_id, session.id)

        # Parse JSON response
        result = json.loads(strip_code_fence(response))

        # Convert to NormalizedEntity objects
        normalized: list[NormalizedEntity] = []
        for item in result["normalized_entities"]:
            normalized.append(NormalizedEntity(
                original_value=item["original_value"],
                normalized_value=item["normalized_value"],
                confidence=item["confidence"],
                reason=item["reason"],
                context=""  # Not used currently
            ))

        # Step 4: Populate cache
        await self._populate_cache(normalized, self.cache)

        # Step 5: Combine results in original order
        all_results: dict[str, NormalizedEntity] = {**cached_results, **{e.original_value: e for e in normalized}}

        # Step 6: Log cache stats
        await self._log_cache_stats(self.cache)

        return [all_results[e] for e in entities]

    async def _call_agent_async(self, query: str, user_id: str, session_id: str) -> str:
        """Call the normalization agent and return the final response."""
        # Prepare the user's message in ADK format
        content: Content = types.Content(
            role='user',
            parts=[types.Part(text=query)]
        )

        final_response_text = "Agent did not produce a final response."

        # Execute the agent and iterate through events
        async for event in self.runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                break

        return final_response_text

    async def _get_cached_entities(
        self,
        entities: list[str],
        cache: EntityCache | None
    ) -> tuple[dict[str, NormalizedEntity], list[str]]:
        """
        Split entities into cached vs uncached.

        Args:
            entities: List of entity names to check
            cache: Cache instance (if None, returns all as uncached)

        Returns:
            Tuple of (cached_results dict, uncached_entities list)
        """
        if not cache:
            return {}, entities

        try:
            cached_results: dict[str, NormalizedEntity] = await cache.get_many(entities)
            uncached_entities: list[str] = [e for e in entities if e not in cached_results]
            logger.info(f"Cache lookup: {len(cached_results)} hits, {len(uncached_entities)} misses")
            return cached_results, uncached_entities
        except Exception as e:
            logger.warning(f"Cache get_many failed: {e}. Falling back to LLM for all entities.")
            return {}, entities

    async def _populate_cache(
        self,
        normalized_entities: list[NormalizedEntity],
        cache: EntityCache | None
    ) -> None:
        """
        Populate cache with newly normalized entities.

        Args:
            normalized_entities: List of normalized entities to cache
            cache: Cache instance (if None, does nothing)
        """
        if not cache or not normalized_entities:
            return

        try:
            cache_entries = {entity.original_value: entity for entity in normalized_entities}
            await cache.set_many(cache_entries)
            logger.info(f"Cached {len(cache_entries)} newly normalized entities")
        except Exception as e:
            logger.warning(f"Cache set_many failed: {e}. Continuing without caching.")

    async def _log_cache_stats(self, cache: EntityCache | None) -> None:
        """
        Log cache performance stats (hit rate).

        Args:
            cache: Cache instance (if None, does nothing)
        """
        if not cache:
            return

        try:
            stats = cache.get_stats()
            logger.info(
                f"Cache stats: hit_rate={stats['hit_rate']:.1%}, "
                f"size={stats['size']:,}/{stats['max_size']:,}, "
                f"hits={stats['hits']}, misses={stats['misses']}"
            )
        except Exception as e:
            logger.debug(f"Failed to log cache stats: {e}")
