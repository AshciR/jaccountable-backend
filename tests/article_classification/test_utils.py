"""Tests for src.article_classification.utils."""
import pytest

from src.article_classification.utils import strip_code_fence


class TestStripCodeFence:
    """Strip Claude's markdown JSON fences before passing to Pydantic.

    Anthropic models wrap their JSON responses in ```json ... ``` by default,
    which causes `model_validate_json` to fail with a json_invalid error.
    OpenAI's response_format=json_object used to strip these for us.
    """

    def test_strips_json_language_fence(self):
        # Given: Response wrapped in ```json ... ``` (the production failure mode)
        response = '```json\n{"is_relevant": true, "confidence": 0.9}\n```'

        # When: Stripping the fence
        result = strip_code_fence(response)

        # Then: Returns raw JSON, ready for model_validate_json
        assert result == '{"is_relevant": true, "confidence": 0.9}'

    def test_strips_plain_fence_without_language_tag(self):
        # Given: Response wrapped in ``` ... ``` (no language hint)
        response = '```\n{"key": "value"}\n```'

        # When: Stripping the fence
        result = strip_code_fence(response)

        # Then: Returns raw JSON
        assert result == '{"key": "value"}'

    def test_returns_raw_json_unchanged_when_no_fence(self):
        # Given: Raw JSON with no markdown wrapping
        response = '{"is_relevant": false, "confidence": 0.2}'

        # When: Stripping
        result = strip_code_fence(response)

        # Then: Returns input unchanged
        assert result == '{"is_relevant": false, "confidence": 0.2}'

    def test_strips_leading_and_trailing_whitespace_around_fence(self):
        # Given: Fence surrounded by whitespace (e.g. trailing newline from LLM)
        response = '\n  ```json\n{"a": 1}\n```  \n'

        # When: Stripping
        result = strip_code_fence(response)

        # Then: Whitespace and fence both removed
        assert result == '{"a": 1}'

    def test_handles_multiline_json_inside_fence(self):
        # Given: Pretty-printed multi-line JSON inside fence (Claude's default)
        response = '```json\n{\n    "is_relevant": true,\n    "confidence": 0.9,\n    "key_entities": ["OCG", "Ruel Reid"]\n}\n```'

        # When: Stripping
        result = strip_code_fence(response)

        # Then: Inner JSON preserved with formatting
        expected = '{\n    "is_relevant": true,\n    "confidence": 0.9,\n    "key_entities": ["OCG", "Ruel Reid"]\n}'
        assert result == expected

    def test_returns_empty_string_for_empty_input(self):
        # Given: Empty response (shouldn't happen but shouldn't crash)
        # When/Then: Returns empty without raising
        assert strip_code_fence("") == ""

    def test_returns_unfenced_text_when_only_one_fence_present(self):
        # Given: Malformed input with only opening fence — not a valid wrapped block
        # When/Then: Treated as raw text; we don't try to be too clever
        response = '```json\n{"a": 1}'

        result = strip_code_fence(response)

        # Then: Returns the raw input stripped of whitespace; downstream
        # parser will surface the JSON error (better than silent partial strip).
        assert result == '```json\n{"a": 1}'
