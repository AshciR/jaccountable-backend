"""Article extractor for Jamaica Gleaner news source (V2 - JSON-LD + CSS hybrid)."""
import base64
import json
from datetime import datetime, timezone
from bs4 import BeautifulSoup

from src.article_extractor.models import ExtractedArticleContent

# schema.org types accepted as the article JSON-LD block. The Gleaner currently
# emits "NewsArticle"; older pages and test fixtures use "Article".
ARTICLE_JSON_LD_TYPES = frozenset({"Article", "NewsArticle", "ReportageNewsArticle"})


class GleanerExtractorV2:
    """
    V2 extraction strategy for Jamaica Gleaner articles (JSON-LD + CSS hybrid).

    This is the current implementation with JSON-LD parsing priority
    and comprehensive CSS fallbacks.

    Implements ArticleExtractor Protocol for jamaica-gleaner.com.

    Uses hybrid parsing approach:
    1. JSON-LD structured data for metadata (title, author, date) - more stable
    2. CSS selectors for article body text (JSON-LD doesn't include full text)
    3. Fallback to legacy CSS selectors for backward compatibility

    HTML Structure (validated 2025-12-10):
    - JSON-LD: <script type="application/ld+json"> with Schema.org Article
    - Article title: headline field in JSON-LD (fallback: h1.article--title, h1.title, any h1)
    - Article content: <div class="article--body"> with <p> tags (fallback: article-content, field-name-body)
    - Author: author.name field in JSON-LD (fallback: div.article--authors, a.author-term)
    - Published date: datePublished field in JSON-LD (fallback: meta[article:published_time], time[datetime])
    """

    def extract(self, html: str, url: str) -> ExtractedArticleContent:
        """
        Extract article content from Gleaner HTML using hybrid JSON-LD + CSS parsing (V2 strategy).

        Args:
            html: Raw HTML content
            url: Article URL (for error context)

        Returns:
            ExtractedArticleContent with extracted data

        Raises:
            ValueError: If required elements (title, full_text) are missing
        """
        soup = BeautifulSoup(html, "lxml")

        # Extract JSON-LD structured data (if available)
        json_ld = self._extract_json_ld(soup)

        # Extract required fields (fail-fast if missing)
        title = self._extract_title(soup, json_ld, url)
        full_text = self._extract_full_text(soup, url)

        # Extract optional fields (return None if missing)
        author = self._extract_author(soup, json_ld)
        published_date = self._extract_published_date(soup, json_ld)

        return ExtractedArticleContent(
            title=title,
            full_text=full_text,
            author=author,
            published_date=published_date,
        )

    def _extract_json_ld(self, soup: BeautifulSoup) -> dict | None:
        """
        Extract and parse JSON-LD structured data from HTML.

        Args:
            soup: BeautifulSoup parsed HTML

        Returns:
            Parsed JSON-LD dict if found and valid, None otherwise
        """
        # Find all script tags with type="application/ld+json"
        json_ld_scripts = soup.find_all("script", type="application/ld+json")

        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                # Look for an Article type (could be multiple JSON-LD blocks).
                # @type may be a string or a list of strings.
                if isinstance(data, dict):
                    json_ld_type = data.get("@type")
                    types = json_ld_type if isinstance(json_ld_type, list) else [json_ld_type]
                    if ARTICLE_JSON_LD_TYPES.intersection(t for t in types if isinstance(t, str)):
                        return data
            except (json.JSONDecodeError, TypeError, AttributeError):
                # Malformed JSON or missing content - continue to next script tag
                continue

        return None

    def _extract_title(self, soup: BeautifulSoup, json_ld: dict | None, url: str) -> str:
        """
        Extract article title with priority fallback chain.

        Priority:
        1. JSON-LD headline field (most reliable)
        2. h1.article--title (new site structure)
        3. h1.title (legacy site structure)
        4. Any h1 tag (last resort)

        Args:
            soup: BeautifulSoup parsed HTML
            json_ld: Parsed JSON-LD data (or None)
            url: Article URL for error context

        Returns:
            Article title string

        Raises:
            ValueError: If title cannot be extracted from any source
        """
        # Priority 1: JSON-LD headline
        if json_ld and "headline" in json_ld:
            headline = json_ld["headline"]
            if isinstance(headline, str) and headline.strip():
                return headline.strip()

        # Priority 2: h1 with class="article--title" (new site)
        title_tag = soup.find("h1", class_="article--title")
        if title_tag:
            title_text = title_tag.get_text(strip=True)
            if title_text:
                return title_text

        # Priority 3: h1 with class="title" (legacy)
        title_tag = soup.find("h1", class_="title")
        if title_tag:
            title_text = title_tag.get_text(strip=True)
            if title_text:
                return title_text

        # Priority 4: Any h1 tag (last resort)
        h1_tag = soup.find("h1")
        if h1_tag:
            title_text = h1_tag.get_text(strip=True)
            if title_text:
                return title_text

        # If no title found from any source, raise error (fail-fast)
        raise ValueError(f"Could not extract title from article: {url}")

    def _is_premium_article(self, soup: BeautifulSoup) -> bool:
        """
        Check if the article is a premium (paywalled) article.

        Premium articles have their content base64-encoded in Drupal settings JSON
        rather than rendered in the DOM, because the Piano.js paywall injects it
        client-side.

        Args:
            soup: BeautifulSoup parsed HTML

        Returns:
            True if the article is marked as premium, False otherwise
        """
        drupal_settings = self._extract_drupal_settings(soup)
        if drupal_settings is None:
            return False
        piano_fields = drupal_settings.get("gleanerPianoFields", {})
        return piano_fields.get("premium") == "1"

    def _extract_drupal_settings(self, soup: BeautifulSoup) -> dict | None:
        """
        Extract and parse the Drupal settings JSON from the page.

        Args:
            soup: BeautifulSoup parsed HTML

        Returns:
            Parsed Drupal settings dict, or None if not found or malformed
        """
        script_tag = soup.find("script", attrs={"data-drupal-selector": "drupal-settings-json"})
        if not script_tag or not script_tag.string:
            return None
        try:
            return json.loads(script_tag.string)
        except (json.JSONDecodeError, TypeError):
            return None

    def _extract_premium_content(self, soup: BeautifulSoup) -> BeautifulSoup | None:
        """
        Extract article content from the base64-encoded premium content blob.

        Premium articles store their rendered HTML body as base64-encoded JSON
        inside the Drupal settings JSON under paywalled_jsonld.premiumContent.

        Args:
            soup: BeautifulSoup parsed HTML

        Returns:
            BeautifulSoup fragment of the article body, or None if extraction fails
        """
        drupal_settings = self._extract_drupal_settings(soup)
        if drupal_settings is None:
            return None

        try:
            b64_content = drupal_settings["paywalled_jsonld"]["premiumContent"]
            decoded = json.loads(base64.b64decode(b64_content))
            rendered_body = decoded["rendered_body"]
            return BeautifulSoup(rendered_body, "lxml")
        except (KeyError, json.JSONDecodeError, TypeError, Exception):
            return None

    def _extract_full_text(self, soup: BeautifulSoup, url: str) -> str:
        """
        Extract article body paragraphs with priority fallback chain.

        For premium articles, content is extracted from the base64-encoded
        premium content blob in Drupal settings JSON.

        For non-premium articles:
        1. div.article--body (new site structure)
        2. div.article-content (legacy site structure)
        3. div.field-name-body (older legacy structure)

        Args:
            soup: BeautifulSoup parsed HTML
            url: Article URL for error context

        Returns:
            Full article text as string

        Raises:
            ValueError: If article body cannot be extracted or is too short
        """
        # Premium articles: content is base64-encoded in Drupal settings JSON
        if self._is_premium_article(soup):
            premium_soup = self._extract_premium_content(soup)
            if premium_soup is None:
                raise ValueError(f"Premium article but could not decode content: {url}")
            return self._extract_paragraphs(premium_soup, url)

        # Non-premium: try CSS selectors in priority order
        content_container = None

        # Priority 1: div.article--body (new site)
        content_container = soup.find("div", class_="article--body")

        # Priority 2: div.article-content (legacy)
        if not content_container:
            content_container = soup.find("div", class_="article-content")

        # Priority 3: div.field-name-body (older legacy)
        if not content_container:
            content_container = soup.find("div", class_="field-name-body")

        if not content_container:
            raise ValueError(f"Could not find article content container: {url}")

        return self._extract_paragraphs(content_container, url)

    def _extract_paragraphs(self, container: BeautifulSoup, url: str) -> str:
        """
        Extract and filter paragraphs from an HTML container.

        Args:
            container: BeautifulSoup element containing <p> tags
            url: Article URL for error context

        Returns:
            Full article text as string

        Raises:
            ValueError: If no valid paragraphs or text too short
        """
        paragraphs = container.find_all("p")

        if not paragraphs:
            raise ValueError(f"No paragraphs found in article content: {url}")

        # Join paragraphs with double newline
        # Filter out empty paragraphs and email addresses (last paragraph often contains reporter email)
        valid_paragraphs = []
        for p in paragraphs:
            text = p.get_text(strip=True)
            # Skip empty paragraphs and email-only paragraphs
            if text and not text.endswith("@gleanerjm.com"):
                valid_paragraphs.append(text)

        full_text = "\n\n".join(valid_paragraphs)

        if not full_text or len(full_text) < 50:
            raise ValueError(f"Extracted text too short or empty: {url}")

        return full_text

    def _extract_author(self, soup: BeautifulSoup, json_ld: dict | None) -> str | None:
        """
        Extract author name with priority fallback chain (optional field).

        Priority:
        1. JSON-LD author.name field (most reliable)
        2. div.article--authors (new site structure)
        3. a.author-term (legacy site structure)

        Args:
            soup: BeautifulSoup parsed HTML
            json_ld: Parsed JSON-LD data (or None)

        Returns:
            Author name string or None if not found
        """
        # Priority 1: JSON-LD author.name
        if json_ld and "author" in json_ld:
            author_obj = json_ld["author"]
            if isinstance(author_obj, dict) and author_obj.get("@type") == "Person":
                author_name = author_obj.get("name")
                if isinstance(author_name, str) and author_name.strip():
                    return self._clean_author_name(author_name.strip())

        # Priority 2: div.article--authors (new site)
        author_div = soup.find("div", class_="article--authors")
        if author_div:
            author_text = author_div.get_text(strip=True)
            if author_text:
                return self._clean_author_name(author_text)

        # Priority 3: a.author-term (legacy)
        author_link = soup.find("a", class_="author-term")
        if author_link:
            author_text = author_link.get_text(strip=True)
            if author_text:
                return self._clean_author_name(author_text)

        return None

    def _extract_published_date(self, soup: BeautifulSoup, json_ld: dict | None) -> datetime | None:
        """
        Extract published date with priority fallback chain (optional field).

        Priority:
        1. JSON-LD datePublished field (most reliable)
        2. meta[property="article:published_time"] (legacy)
        3. time[datetime] tag (older legacy)

        Args:
            soup: BeautifulSoup parsed HTML
            json_ld: Parsed JSON-LD data (or None)

        Returns:
            Timezone-aware datetime in UTC or None if not found
        """
        # Priority 1: JSON-LD datePublished
        if json_ld and "datePublished" in json_ld:
            date_str = json_ld["datePublished"]
            if isinstance(date_str, str):
                parsed_date = self._parse_and_normalize_date(date_str)
                if parsed_date:
                    return parsed_date

        # Priority 2: meta tag with property="article:published_time"
        meta_date = soup.find("meta", property="article:published_time")
        if meta_date:
            content = meta_date.get("content")
            if content:
                parsed_date = self._parse_and_normalize_date(content)
                if parsed_date:
                    return parsed_date

        # Priority 3: time tag with datetime attribute
        time_tag = soup.find("time")
        if time_tag:
            datetime_attr = time_tag.get("datetime")
            if datetime_attr:
                parsed_date = self._parse_and_normalize_date(datetime_attr)
                if parsed_date:
                    return parsed_date

        return None

    def _clean_author_name(self, author_text: str) -> str:
        """
        Clean up author name by removing common prefixes and suffixes.

        Removes:
        - "By " / "by " prefix
        - "/Staff Reporter" suffix
        - "/Senior Staff Reporter" suffix
        - "/Gleaner Writer" suffix (and other variants)

        Args:
            author_text: Raw author name string

        Returns:
            Cleaned author name string
        """
        author_text = author_text.replace("By ", "").replace("by ", "")
        # Remove all variants of "/Staff Reporter" and similar suffixes
        # Split on "/" and take only the first part (the actual name)
        if "/" in author_text:
            author_text = author_text.split("/")[0].strip()
        return author_text

    def _parse_and_normalize_date(self, date_str: str) -> datetime | None:
        """
        Parse ISO 8601 date string and normalize to timezone-aware UTC datetime.

        Args:
            date_str: ISO 8601 formatted date string

        Returns:
            Timezone-aware datetime in UTC or None if parsing fails
        """
        try:
            dt = datetime.fromisoformat(date_str)
            # Ensure timezone-aware (convert to UTC)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except (ValueError, TypeError):
            return None
