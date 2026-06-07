"""External service contract test for GleanerArchiveExtractor.

This test validates that the Gleaner newspaper archive website structure hasn't changed
in ways that would break our extractor. It makes a live HTTP request to verify
the current archive site structure matches our implementation expectations.

Run with: uv run pytest tests/article_extractor/extractors/test_gleaner_archive_extractor_contract.py -m contract -v
"""
from datetime import datetime, timezone

import httpx
import pytest

from src.article_extractor.extractors.gleaner_archive_extractor import GleanerArchiveExtractor


class TestGleanerArchiveExtractorExternalContract:
    """External contract test that validates live archive site structure."""

    @pytest.mark.external
    @pytest.mark.contract
    async def test_gleaner_archive_site_structure_unchanged(self):
        """
        Verify Gleaner archive site structure works with extractor.

        This test detects breaking changes to the gleaner.newspaperarchive.com website
        by validating that our extractor can successfully parse a known live archive page.

        The test validates:
        - Extractor successfully extracts OCR content from live archive page
        - Title extraction works (via LLM or fallback to og:title)
        - Author extraction works (via LLM from OCR text)
        - Date extraction works (from URL pattern)
        - Full text extraction works (from OCR sections)

        Note: This test makes real LLM API calls and requires OPENAI_EXTRACTOR_API_KEY.
        """
        # Given: Known live Gleaner archive page (Nov 7, 2021, page 5 - Ruel Reid article)
        url = "https://gleaner.newspaperarchive.com/kingston-gleaner/2021-11-07/page-5/"

        # When: Fetching and extracting content from live archive site
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                timeout=30,
                headers={"User-Agent": "Mozilla/5.0 (compatible; JaacountableBot/1.0)"},
            )
            response.raise_for_status()
            html = response.text

        extractor = GleanerArchiveExtractor()
        content = extractor.extract(html, url)

        # Then: Extraction succeeds with expected values

        # Title should match og:title format (LLM returns NONE, falls back to og:title)
        assert content.title == "Kingston Gleaner Newspaper Archives | Nov 07, 2021, p. 5"

        # Full text should be extracted from OCR sections (about Ruel Reid/CMU case)
        assert content.full_text is not None
        assert len(content.full_text) >= 50
        assert "Reid" in content.full_text or "CMU" in content.full_text

        # Published date should be extracted from URL
        assert content.published_date == datetime(2021, 11, 7, tzinfo=timezone.utc)

        # Author should be extracted by LLM from OCR text
        assert content.author == "Livern Barrett"
