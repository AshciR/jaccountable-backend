"""Tests for GleanerExtractorV2 (JSON-LD + CSS hybrid strategy)."""
import base64
import json
import pytest
from datetime import timezone

from src.article_extractor.extractors.gleaner_extractor_v2 import GleanerExtractorV2
from src.article_extractor.models import ExtractedArticleContent


class TestGleanerExtractorV2HappyPath:
    """Happy path tests for V2 JSON-LD + CSS hybrid extraction."""

    async def test_extract_complete_article(self, gleaner_html_v2: str):
        # Given: valid Gleaner article HTML with JSON-LD and all elements
        extractor = GleanerExtractorV2()
        url = "https://jamaica-gleaner.com/article/news/20251210/embrace-one-health"

        # When: extracting content
        content = extractor.extract(gleaner_html_v2, url)

        # Then: all fields are extracted correctly from JSON-LD and CSS
        assert isinstance(content, ExtractedArticleContent)

        # Title from JSON-LD (with curly quotes U+2018/U+2019 and trailing space stripped)
        assert content.title == "Embrace \u2018One Health\u2019"

        # Full text from CSS (article--body)
        assert content.full_text is not None
        assert content.full_text.startswith("Medical experts are calling for stronger adherence to the global One Health mandate")

        # Author from JSON-LD (cleaned)
        assert content.author == "Corey Robinson"

        # Published date from JSON-LD (converted to UTC)
        assert content.published_date is not None
        assert content.published_date.year == 2025
        assert content.published_date.month == 12
        assert content.published_date.day == 10
        assert content.published_date.tzinfo == timezone.utc

    async def test_author_name_cleaned(self, gleaner_html_v2: str):
        # Given: Gleaner article HTML with author in format "Name/Staff Reporter"
        extractor = GleanerExtractorV2()
        url = "https://jamaica-gleaner.com/article/news/20251210/embrace-one-health"

        # When: extracting content
        content = extractor.extract(gleaner_html_v2, url)

        # Then: "/Staff Reporter" suffix is removed from author name
        assert content.author == "Corey Robinson"
        assert "/Staff Reporter" not in content.author
        assert "By " not in content.author

    async def test_json_ld_missing_falls_back_to_css(self):
        # Given: HTML without JSON-LD but with CSS selectors (new site structure)
        html = """
        <html>
            <body>
                <h1 class="article--title">Test Article Title</h1>
                <div class="article--body">
                    <p>First paragraph of content that is long enough to pass validation.</p>
                    <p>Second paragraph with more content.</p>
                </div>
                <div class="article--authors">Jane Doe/Staff Reporter</div>
                <meta property="article:published_time" content="2025-12-10T12:00:00-05:00">
            </body>
        </html>
        """
        extractor = GleanerExtractorV2()
        url = "https://jamaica-gleaner.com/article/news/test"

        # When: extracting content
        content = extractor.extract(html, url)

        # Then: extraction succeeds using CSS fallbacks
        assert content.title == "Test Article Title"
        assert content.full_text.startswith("First paragraph of content")
        assert content.author == "Jane Doe"
        assert content.published_date is not None

    async def test_json_ld_malformed_falls_back_to_css(self):
        # Given: HTML with malformed JSON-LD but valid CSS selectors
        html = """
        <html>
            <head>
                <script type="application/ld+json">
                    { invalid json here }
                </script>
            </head>
            <body>
                <h1 class="article--title">Fallback Title</h1>
                <div class="article--body">
                    <p>Content paragraph that is long enough to meet minimum requirements for extraction.</p>
                </div>
            </body>
        </html>
        """
        extractor = GleanerExtractorV2()
        url = "https://jamaica-gleaner.com/article/news/test"

        # When: extracting content (should not crash)
        content = extractor.extract(html, url)

        # Then: extraction succeeds using CSS fallbacks
        assert content.title == "Fallback Title"
        assert content.full_text.startswith("Content paragraph")

    async def test_fallback_to_legacy_css_selectors_works(self):
        # Given: HTML with legacy CSS classes (no JSON-LD, no new classes)
        html = """
        <html>
            <body>
                <h1 class="title">Legacy Title</h1>
                <div class="article-content">
                    <p>Legacy content paragraph with enough text to pass validation requirements.</p>
                    <p>Second paragraph of legacy content.</p>
                </div>
                <a class="author-term">Legacy Author/Staff Reporter</a>
            </body>
        </html>
        """
        extractor = GleanerExtractorV2()
        url = "https://jamaica-gleaner.com/article/news/legacy"

        # When: extracting content
        content = extractor.extract(html, url)

        # Then: extraction succeeds using legacy CSS fallbacks
        assert content.title == "Legacy Title"
        assert content.full_text.startswith("Legacy content paragraph")
        assert content.author == "Legacy Author"

    async def test_mixed_json_ld_and_css(self):
        # Given: HTML with JSON-LD for metadata but CSS for body
        html = """
        <html>
            <head>
                <script type="application/ld+json">
                {
                    "@context": "https://schema.org",
                    "@type": "Article",
                    "headline": "Mixed Source Title",
                    "author": {
                        "@type": "Person",
                        "name": "Mixed Author"
                    },
                    "datePublished": "2025-12-10T10:00:00-05:00"
                }
                </script>
            </head>
            <body>
                <div class="article--body">
                    <p>Body content from CSS selector with sufficient length for validation.</p>
                </div>
            </body>
        </html>
        """
        extractor = GleanerExtractorV2()
        url = "https://jamaica-gleaner.com/article/news/mixed"

        # When: extracting content
        content = extractor.extract(html, url)

        # Then: metadata from JSON-LD, body from CSS
        assert content.title == "Mixed Source Title"
        assert content.author == "Mixed Author"
        assert content.published_date is not None
        assert content.full_text.startswith("Body content from CSS")


class TestGleanerExtractorV2ParsingErrors:
    """V2 parsing error tests."""

    async def test_missing_title_all_sources_raises_value_error(self):
        # Given: HTML without title in JSON-LD or any CSS selector
        html = """
        <html>
            <head>
                <script type="application/ld+json">
                {
                    "@context": "https://schema.org",
                    "@type": "Article",
                    "author": {"@type": "Person", "name": "Author"}
                }
                </script>
            </head>
            <body>
                <div class="article--body"><p>Content without title</p></div>
            </body>
        </html>
        """
        extractor = GleanerExtractorV2()
        url = "https://jamaica-gleaner.com/article/news/test"

        # When/Then: extraction raises ValueError
        with pytest.raises(ValueError) as exc_info:
            extractor.extract(html, url)

        assert "Could not extract title" in str(exc_info.value)
        assert url in str(exc_info.value)

    async def test_missing_content_all_sources_raises_value_error(self):
        # Given: HTML without article content container in any CSS selector
        html = """
        <html>
            <head>
                <script type="application/ld+json">
                {
                    "@context": "https://schema.org",
                    "@type": "Article",
                    "headline": "Title Only"
                }
                </script>
            </head>
            <body>
                <h1>Title Only</h1>
            </body>
        </html>
        """
        extractor = GleanerExtractorV2()
        url = "https://jamaica-gleaner.com/article/news/test"

        # When/Then: extraction raises ValueError
        with pytest.raises(ValueError) as exc_info:
            extractor.extract(html, url)

        assert "content container" in str(exc_info.value).lower()
        assert url in str(exc_info.value)

    async def test_empty_content_div_raises_value_error(self):
        # Given: HTML with content container but no paragraphs
        html = """
        <html>
            <body>
                <h1 class="article--title">Test Title</h1>
                <div class="article--body">
                    <div>Some div content but no paragraphs</div>
                </div>
            </body>
        </html>
        """
        extractor = GleanerExtractorV2()
        url = "https://jamaica-gleaner.com/article/news/test"

        # When/Then: extraction raises ValueError
        with pytest.raises(ValueError) as exc_info:
            extractor.extract(html, url)

        assert "No paragraphs found" in str(exc_info.value)
        assert url in str(exc_info.value)

    async def test_too_short_text_raises_value_error(self):
        # Given: HTML with content that is too short (< 50 characters)
        html = """
        <html>
            <body>
                <h1 class="article--title">Title</h1>
                <div class="article--body">
                    <p>Short</p>
                </div>
            </body>
        </html>
        """
        extractor = GleanerExtractorV2()
        url = "https://jamaica-gleaner.com/article/news/test"

        # When/Then: extraction raises ValueError
        with pytest.raises(ValueError) as exc_info:
            extractor.extract(html, url)

        assert "too short" in str(exc_info.value).lower()
        assert url in str(exc_info.value)

    async def test_json_ld_invalid_json_gracefully_degrades(self):
        # Given: HTML with invalid JSON in JSON-LD script tag
        html = """
        <html>
            <head>
                <script type="application/ld+json">
                    { "invalid": json, missing: quotes }
                </script>
            </head>
            <body>
                <h1 class="article--title">Fallback Title</h1>
                <div class="article--body">
                    <p>Content that will be extracted despite invalid JSON-LD with enough length.</p>
                </div>
            </body>
        </html>
        """
        extractor = GleanerExtractorV2()
        url = "https://jamaica-gleaner.com/article/news/test"

        # When: extracting content (should not crash)
        content = extractor.extract(html, url)

        # Then: extraction succeeds using CSS fallbacks
        assert content.title == "Fallback Title"
        assert content.full_text.startswith("Content that will be extracted")


class TestGleanerExtractorV2EdgeCases:
    """V2 edge case tests."""

    async def test_missing_author_returns_none(self):
        # Given: HTML without author in JSON-LD or any CSS selector
        html = """
        <html>
            <head>
                <script type="application/ld+json">
                {
                    "@context": "https://schema.org",
                    "@type": "Article",
                    "headline": "No Author Article"
                }
                </script>
            </head>
            <body>
                <div class="article--body">
                    <p>Article content without author information that meets length requirements.</p>
                </div>
            </body>
        </html>
        """
        extractor = GleanerExtractorV2()
        url = "https://jamaica-gleaner.com/article/news/test"

        # When: extracting content
        content = extractor.extract(html, url)

        # Then: author is None (optional field)
        assert content.author is None

    async def test_missing_date_returns_none(self):
        # Given: HTML without date in JSON-LD or any CSS selector
        html = """
        <html>
            <head>
                <script type="application/ld+json">
                {
                    "@context": "https://schema.org",
                    "@type": "Article",
                    "headline": "No Date Article"
                }
                </script>
            </head>
            <body>
                <div class="article--body">
                    <p>Article content without publication date that has sufficient length.</p>
                </div>
            </body>
        </html>
        """
        extractor = GleanerExtractorV2()
        url = "https://jamaica-gleaner.com/article/news/test"

        # When: extracting content
        content = extractor.extract(html, url)

        # Then: published_date is None (optional field)
        assert content.published_date is None

    async def test_json_ld_missing_headline_falls_back(self):
        # Given: JSON-LD without headline field
        html = """
        <html>
            <head>
                <script type="application/ld+json">
                {
                    "@context": "https://schema.org",
                    "@type": "Article",
                    "author": {"@type": "Person", "name": "Someone"}
                }
                </script>
            </head>
            <body>
                <h1 class="article--title">CSS Fallback Title</h1>
                <div class="article--body">
                    <p>Content extracted from CSS with adequate length for validation.</p>
                </div>
            </body>
        </html>
        """
        extractor = GleanerExtractorV2()
        url = "https://jamaica-gleaner.com/article/news/test"

        # When: extracting content
        content = extractor.extract(html, url)

        # Then: title extracted from CSS fallback
        assert content.title == "CSS Fallback Title"

    async def test_json_ld_author_not_person_type(self):
        # Given: JSON-LD with author but not Person type
        html = """
        <html>
            <head>
                <script type="application/ld+json">
                {
                    "@context": "https://schema.org",
                    "@type": "Article",
                    "headline": "Article Title",
                    "author": {
                        "@type": "Organization",
                        "name": "News Org"
                    }
                }
                </script>
            </head>
            <body>
                <div class="article--body">
                    <p>Content with organization author instead of person with sufficient length.</p>
                </div>
                <div class="article--authors">Fallback Author</div>
            </body>
        </html>
        """
        extractor = GleanerExtractorV2()
        url = "https://jamaica-gleaner.com/article/news/test"

        # When: extracting content
        content = extractor.extract(html, url)

        # Then: falls back to CSS selector for author
        assert content.author == "Fallback Author"

    async def test_date_parsing_error_returns_none(self):
        # Given: JSON-LD with invalid date format
        html = """
        <html>
            <head>
                <script type="application/ld+json">
                {
                    "@context": "https://schema.org",
                    "@type": "Article",
                    "headline": "Article",
                    "datePublished": "not-a-valid-date"
                }
                </script>
            </head>
            <body>
                <div class="article--body">
                    <p>Content with invalid date that still meets length requirements.</p>
                </div>
            </body>
        </html>
        """
        extractor = GleanerExtractorV2()
        url = "https://jamaica-gleaner.com/article/news/test"

        # When: extracting content
        content = extractor.extract(html, url)

        # Then: published_date is None (graceful degradation)
        assert content.published_date is None

    async def test_unicode_content_in_json_ld(self):
        # Given: JSON-LD with Unicode characters in headline and author
        html = """
        <html>
            <head>
                <script type="application/ld+json">
                {
                    "@context": "https://schema.org",
                    "@type": "Article",
                    "headline": "Café Résumé: A Story",
                    "author": {
                        "@type": "Person",
                        "name": "José García"
                    }
                }
                </script>
            </head>
            <body>
                <div class="article--body">
                    <p>Content with Unicode characters: café, résumé, naïve with sufficient length.</p>
                </div>
            </body>
        </html>
        """
        extractor = GleanerExtractorV2()
        url = "https://jamaica-gleaner.com/article/news/test"

        # When: extracting content
        content = extractor.extract(html, url)

        # Then: Unicode characters preserved correctly
        assert content.title == "Café Résumé: A Story"
        assert content.author == "José García"


class TestGleanerExtractorV2JsonLdParsing:
    """Tests specifically for V2 JSON-LD parsing logic."""

    async def test_extract_json_ld_success(self):
        # Given: HTML with valid JSON-LD Article
        html = """
        <html>
            <head>
                <script type="application/ld+json">
                {
                    "@context": "https://schema.org",
                    "@type": "Article",
                    "headline": "Test"
                }
                </script>
            </head>
        </html>
        """
        extractor = GleanerExtractorV2()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")

        # When: extracting JSON-LD
        json_ld = extractor._extract_json_ld(soup)

        # Then: JSON-LD is parsed correctly
        assert json_ld is not None
        assert json_ld["@type"] == "Article"
        assert json_ld["headline"] == "Test"

    async def test_extract_json_ld_missing_returns_none(self):
        # Given: HTML without JSON-LD script tag
        html = """
        <html>
            <head>
                <title>No JSON-LD</title>
            </head>
        </html>
        """
        extractor = GleanerExtractorV2()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")

        # When: extracting JSON-LD
        json_ld = extractor._extract_json_ld(soup)

        # Then: returns None
        assert json_ld is None

    async def test_extract_json_ld_invalid_json_returns_none(self):
        # Given: HTML with invalid JSON in script tag
        html = """
        <html>
            <head>
                <script type="application/ld+json">
                    { invalid json }
                </script>
            </head>
        </html>
        """
        extractor = GleanerExtractorV2()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")

        # When: extracting JSON-LD
        json_ld = extractor._extract_json_ld(soup)

        # Then: returns None (graceful error handling)
        assert json_ld is None

    async def test_extract_json_ld_multiple_scripts_finds_article(self):
        # Given: HTML with multiple JSON-LD blocks, one is Article type
        html = """
        <html>
            <head>
                <script type="application/ld+json">
                {
                    "@context": "https://schema.org",
                    "@type": "WebSite",
                    "name": "Test Site"
                }
                </script>
                <script type="application/ld+json">
                {
                    "@context": "https://schema.org",
                    "@type": "Article",
                    "headline": "The Article"
                }
                </script>
            </head>
        </html>
        """
        extractor = GleanerExtractorV2()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")

        # When: extracting JSON-LD
        json_ld = extractor._extract_json_ld(soup)

        # Then: finds the Article type JSON-LD
        assert json_ld is not None
        assert json_ld["@type"] == "Article"
        assert json_ld["headline"] == "The Article"

    async def test_news_article_type_extracts_published_date(self):
        # Given: HTML with JSON-LD typed "NewsArticle" (current Gleaner markup)
        # and no meta/time date fallbacks
        html = """
        <html>
            <head>
                <script type="application/ld+json">
                {
                    "@context": "https://schema.org",
                    "@type": "NewsArticle",
                    "headline": "Mining ministry moves to overhaul regulation after audit findings",
                    "datePublished": "2026-09-25T01:06:07-04:00"
                }
                </script>
            </head>
            <body>
                <div class="article--body">
                    <p>Article content with sufficient length to pass the validation check.</p>
                </div>
            </body>
        </html>
        """
        extractor = GleanerExtractorV2()
        url = "https://jamaica-gleaner.com/article/news/20260925/mining-ministry-moves-overhaul-regulation-after-audit-findings"

        # When: extracting content
        content = extractor.extract(html, url)

        # Then: published_date comes from JSON-LD, normalized to UTC
        assert content.published_date is not None
        assert content.published_date.tzinfo == timezone.utc
        assert content.published_date.isoformat() == "2026-09-25T05:06:07+00:00"

    async def test_extract_json_ld_type_list_finds_article(self):
        # Given: HTML with JSON-LD whose @type is a list
        html = """
        <html>
            <head>
                <script type="application/ld+json">
                {
                    "@context": "https://schema.org",
                    "@type": ["NewsArticle", "CreativeWork"],
                    "headline": "Listed Type"
                }
                </script>
            </head>
        </html>
        """
        extractor = GleanerExtractorV2()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")

        # When: extracting JSON-LD
        json_ld = extractor._extract_json_ld(soup)

        # Then: the block is recognized as an article
        assert json_ld is not None
        assert json_ld["headline"] == "Listed Type"


class TestGleanerExtractorV2HtmlEntityDecoding:
    """HTML entity decoding in extracted titles."""

    async def test_title_with_html_entities_decoded_via_json_ld(self):
        # Given: JSON-LD headline containing HTML entities (json.loads does not decode these)
        html = """
        <html>
            <head>
                <script type="application/ld+json">
                {
                    "@type": "Article",
                    "headline": "Gleaner&#8217;s report on &#8216;housing crisis&#8217;"
                }
                </script>
            </head>
            <body>
                <div class="article--body">
                    <p>Article content that is long enough to pass validation checks here.</p>
                    <p>Second paragraph to ensure minimum content length requirements are met.</p>
                </div>
            </body>
        </html>
        """
        extractor = GleanerExtractorV2()
        url = "https://jamaica-gleaner.com/article/news/test"

        # When: extracting content
        content = extractor.extract(html, url)

        # Then: HTML entities in the JSON-LD headline are decoded to Unicode
        assert content.title == "Gleaner\u2019s report on \u2018housing crisis\u2019"


class TestGleanerExtractorV2PremiumArticles:
    """Tests for premium (paywalled) article extraction via base64-encoded content."""

    async def test_extract_premium_article_full(self, gleaner_html_v2_premium: str):
        # Given: premium Gleaner article with content in paywalled_jsonld
        extractor = GleanerExtractorV2()
        url = "https://jamaica-gleaner.com/article/news/20260426/nswma-hiring-row-sparks-two-year-tension-between-senior-officials"

        # When: extracting content
        content = extractor.extract(gleaner_html_v2_premium, url)

        # Then: all fields are extracted correctly
        assert isinstance(content, ExtractedArticleContent)
        assert content.title == "NSWMA hiring row sparks two-year tension between senior officials"
        assert content.full_text.startswith("An \u201cirregular\u201d recruitment exercise")
        assert "National Solid Waste Management Authority" in content.full_text
        assert content.author == "Kimone Francis - Senior Staff Reporter"
        assert content.published_date is not None
        assert content.published_date.year == 2026
        assert content.published_date.month == 4
        assert content.published_date.day == 26

    async def test_premium_article_filters_email_paragraphs(self, gleaner_html_v2_premium: str):
        # Given: premium article with email paragraph in rendered_body
        extractor = GleanerExtractorV2()
        url = "https://jamaica-gleaner.com/article/news/test"

        # When: extracting content
        content = extractor.extract(gleaner_html_v2_premium, url)

        # Then: email paragraph is filtered out
        assert "kimone.francis@gleanerjm.com" not in content.full_text

    async def test_premium_article_with_malformed_base64_raises_value_error(self):
        # Given: premium article with invalid base64 in premiumContent
        html = """
        <html>
            <head>
                <script type="application/ld+json">
                {"@type": "Article", "headline": "Test Premium"}
                </script>
                <script type="application/json" data-drupal-selector="drupal-settings-json">
                {"gleanerPianoFields":{"premium":"1"},"paywalled_jsonld":{"premiumContent":"!!!not-valid-base64!!!"}}
                </script>
            </head>
            <body><h1 class="article--title">Test Premium</h1></body>
        </html>
        """
        extractor = GleanerExtractorV2()
        url = "https://jamaica-gleaner.com/article/news/test"

        # When/Then: extraction raises ValueError
        with pytest.raises(ValueError) as exc_info:
            extractor.extract(html, url)

        assert "Premium article but could not decode content" in str(exc_info.value)

    async def test_premium_article_with_missing_rendered_body_raises_value_error(self):
        # Given: premium article with base64 content missing rendered_body key
        bad_content = base64.b64encode(json.dumps({"nid": "123"}).encode()).decode()
        html = f"""
        <html>
            <head>
                <script type="application/ld+json">
                {{"@type": "Article", "headline": "Test Premium"}}
                </script>
                <script type="application/json" data-drupal-selector="drupal-settings-json">
                {{"gleanerPianoFields":{{"premium":"1"}},"paywalled_jsonld":{{"premiumContent":"{bad_content}"}}}}
                </script>
            </head>
            <body><h1 class="article--title">Test Premium</h1></body>
        </html>
        """
        extractor = GleanerExtractorV2()
        url = "https://jamaica-gleaner.com/article/news/test"

        # When/Then: extraction raises ValueError
        with pytest.raises(ValueError) as exc_info:
            extractor.extract(html, url)

        assert "Premium article but could not decode content" in str(exc_info.value)

    async def test_non_premium_article_uses_css_selectors(self):
        # Given: non-premium article with both CSS body and paywalled_jsonld present
        html = """
        <html>
            <head>
                <script type="application/ld+json">
                {"@type": "Article", "headline": "Non Premium Article"}
                </script>
                <script type="application/json" data-drupal-selector="drupal-settings-json">
                {"gleanerPianoFields":{"premium":"0"}}
                </script>
            </head>
            <body>
                <div class="article--body">
                    <p>This content comes from the normal CSS selector path, not premium decoding.</p>
                </div>
            </body>
        </html>
        """
        extractor = GleanerExtractorV2()
        url = "https://jamaica-gleaner.com/article/news/test"

        # When: extracting content
        content = extractor.extract(html, url)

        # Then: content extracted from CSS selectors, not premium path
        assert content.full_text.startswith("This content comes from the normal CSS selector")
