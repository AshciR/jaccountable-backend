"""Integration tests for entity normalization agent.

These tests make actual LLM API calls to verify the normalization agent works correctly.
Run sparingly to avoid API costs.
"""
import pytest
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types
from google.genai.types import Content

from src.article_classification.agents.normalization_agent import normalization_agent
from src.article_classification.base import APP_NAME, NORMALIZATION_MODEL
from src.article_classification.models import (
    EntityNormalizationResult,
)
from src.article_classification.utils import strip_code_fence


@pytest.fixture
def session_service() -> InMemorySessionService:
    """Create session service for agent."""
    return InMemorySessionService()


@pytest.fixture
async def session(session_service: InMemorySessionService) -> Session:
    """Create test session."""
    return await session_service.create_session(
        app_name=APP_NAME,
        user_id="test_normalization"
    )


@pytest.fixture
def runner(session_service: InMemorySessionService) -> Runner:
    """Create runner with normalization agent."""
    return Runner(
        app_name=APP_NAME,
        agent=normalization_agent,
        session_service=session_service
    )


def build_normalization_query(entity_names: list[str], context: str = "") -> str:
    """
    Build a normalization query string for the agent.

    Args:
        entity_names: List of entity names to normalize
        context: Optional context about the entities

    Returns:
        Formatted query string
    """
    context_line = f"Context: {context}\n" if context else ""
    return f"""Please normalize these entity names:

Entity names: {entity_names}
{context_line}
Return the normalization as JSON."""


async def call_agent_async(query: str, runner: Runner, user_id: str, session_id: str) -> str:
    """
    Call the normalization agent and extract final response.

    This is similar to the _call_agent_async pattern used in corruption_classifier.
    """
    content: Content = types.Content(
        role='user',
        parts=[types.Part(text=query)]
    )

    final_response_text = "Agent did not produce a final response."

    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
            break

    return final_response_text


@pytest.mark.external
@pytest.mark.integration
class TestNormalizationAgentIntegration:
    """Integration tests with actual LLM API calls."""

    async def test_normalize_jamaican_official_with_title(
        self, runner: Runner, session: Session
    ):
        # Given: Query with entity to normalize
        query = build_normalization_query(
            entity_names=["Hon. Ruel Reid"],
            context="corruption investigation"
        )

        # When: Normalizing via agent
        response = await call_agent_async(query, runner, session.user_id, session.id)

        # Then: Returns normalized entity with underscores
        result = EntityNormalizationResult.model_validate_json(strip_code_fence(response))
        assert isinstance(result, EntityNormalizationResult)
        assert len(result.normalized_entities) == 1
        entity = result.normalized_entities[0]
        assert entity.original_value == "Hon. Ruel Reid"
        # Should remove title, lowercase, and use underscores
        assert entity.normalized_value == "ruel_reid"
        assert entity.confidence >= 0.8  # Should be confident
        assert len(entity.reason) > 0  # Should have reasoning
        assert result.model_name == NORMALIZATION_MODEL

    async def test_normalize_acronym(
        self, runner: Runner, session: Session
    ):
        # Given: Query with acronym
        query = build_normalization_query(
            entity_names=["OCG"],
            context="government investigation"
        )

        # When: Normalizing via agent
        response = await call_agent_async(query, runner, session.user_id, session.id)

        # Then: Returns acronym in lowercase
        result = EntityNormalizationResult.model_validate_json(strip_code_fence(response))
        assert len(result.normalized_entities) == 1
        entity = result.normalized_entities[0]
        assert entity.original_value == "OCG"
        assert entity.normalized_value == "ocg"
        assert entity.confidence >= 0.9  # Very confident for acronyms
        assert result.model_name == NORMALIZATION_MODEL

    async def test_normalize_ministry(
        self, runner: Runner, session: Session
    ):
        # Given: Query with ministry name
        query = build_normalization_query(
            entity_names=["Ministry of Education"],
            context="government entity"
        )

        # When: Normalizing via agent
        response = await call_agent_async(query, runner, session.user_id, session.id)

        # Then: Returns standardized ministry name with underscores
        result = EntityNormalizationResult.model_validate_json(strip_code_fence(response))
        assert len(result.normalized_entities) == 1
        entity = result.normalized_entities[0]
        assert entity.original_value == "Ministry of Education"
        assert entity.normalized_value == "ministry_of_education"
        assert entity.confidence >= 0.8
        assert result.model_name == NORMALIZATION_MODEL

    async def test_normalize_multiple_entities(
        self, runner: Runner, session: Session
    ):
        # Given: Query with multiple entities
        query = build_normalization_query(
            entity_names=["Hon. Ruel Reid", "OCG", "Ministry of Education", "The OCG"],
            context="corruption investigation involving government officials"
        )

        # When: Normalizing via agent
        response = await call_agent_async(query, runner, session.user_id, session.id)

        # Then: All entities normalized correctly
        result = EntityNormalizationResult.model_validate_json(strip_code_fence(response))
        assert len(result.normalized_entities) == 4

        # Build a dict for easier lookup (original_value -> normalized_value)
        entity_map = {e.original_value: e.normalized_value for e in result.normalized_entities}

        # Check individual normalizations
        assert entity_map["Hon. Ruel Reid"] == "ruel_reid"
        assert entity_map["OCG"] == "ocg"
        assert entity_map["Ministry of Education"] == "ministry_of_education"
        assert entity_map["The OCG"] == "ocg"  # Should handle "The" prefix

        # Each entity should have confidence >= 0.7 and a reason
        for entity in result.normalized_entities:
            assert entity.confidence >= 0.7
            assert len(entity.reason) > 0

        assert result.model_name == NORMALIZATION_MODEL

    async def test_normalize_with_various_titles(
        self, runner: Runner, session: Session
    ):
        # Given: Query with various title variations
        query = build_normalization_query(
            entity_names=["Mr. Andrew Holness", "Dr. Nigel Clarke", "Education Minister Reid", "Prime Minister Holness"]
        )

        # When: Normalizing via agent
        response = await call_agent_async(query, runner, session.user_id, session.id)

        # Then: All titles removed, names normalized with underscores
        result = EntityNormalizationResult.model_validate_json(strip_code_fence(response))
        assert len(result.normalized_entities) == 4

        # Build a dict for easier lookup
        entity_map = {e.original_value: e.normalized_value for e in result.normalized_entities}

        # All should be lowercase with underscores, titles removed
        assert "andrew_holness" in entity_map["Mr. Andrew Holness"]
        assert "nigel_clarke" in entity_map["Dr. Nigel Clarke"]
        assert "reid" in entity_map["Education Minister Reid"]
        assert "holness" in entity_map["Prime Minister Holness"]

        # Each entity should be confident (0.5 floor for ambiguous inputs like
        # "Education Minister Reid" where the first name is missing)
        for entity in result.normalized_entities:
            assert entity.confidence >= 0.5

        assert result.model_name == NORMALIZATION_MODEL
