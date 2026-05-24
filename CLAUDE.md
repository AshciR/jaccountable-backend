# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python backend project called "jaacountable-backend" that implements a news-gathering system focused on Jamaica Gleaner newspaper. The project uses Google ADK (Agent Development Kit) to create LLM agents that scrape and analyze news articles for government accountability topics.

## Key Architecture

### Core Components

- **Main Entry Point**: `main.py` - Simple entry point with basic "Hello World" functionality
- **Article Discovery**: `src/article_discovery/` - RSS-based article discovery system
  - `discoverers/gleaner_rss_discoverer.py` - Multi-feed RSS discoverer for Jamaica Gleaner
  - `models.py` - Discovery models (`RssFeedConfig`, `DiscoveredArticle`)
  - `base.py` - Protocol definition for article discovery
- **Agent System**: `gleaner_researcher_agent/` - Contains the LLM agent implementation
  - `agent.py` - Defines the `news_gatherer_agent` using Google ADK's LlmAgent with LiteLLM model (o4-mini)
  - `tools.py` - Web scraping tools for Jamaica Gleaner sections (lead stories and news)
  - `v1.evalset.json` - Evaluation dataset for testing agent performance
- **Database Layer**: PostgreSQL-based persistence layer
  - `config/database.py` - asyncpg connection pool manager
  - `alembic/` - Database migration system with manual SQL DDL
  - `alembic/versions/` - Migration files with raw SQL
  - `scripts/` - Helper scripts for database operations

### Analytics System

The project uses PostHog for server-side event tracking via a thin `AnalyticsClient` wrapper.

**Key files:**
- `src/analytics/client.py` — `AnalyticsClient` class + module-level `analytics_client` instance
- `src/server/dependencies.py` — `get_analytics(request)` FastAPI dependency
- `src/server/app.py` — wires `analytics_client` into `app.state` at startup and calls `shutdown()` at teardown

**Event naming conventions:**
- Event names: `category:object_action` pattern, lowercase snake_case (e.g., `search:query_submit`)
- Property names: `object_adjective` pattern (e.g., `search_query`, `results_count`)
- Boolean properties: `is_` or `has_` prefix (e.g., `is_internal`, `has_results`)

**Common properties** — every event automatically gets these via `capture_with_common_props()`:
- `environment` — from `APP_ENV` env var (default: `"development"`)
- `is_internal` — from `analytics.is_internal_request(request)` (detects `X-Internal-Request: true` header)

**How to add a new event:**
```python
analytics.capture_with_common_props(
    distinct_id=analytics.get_distinct_id(request),
    event="category:object_action",
    properties={"property_name": value},
    is_internal=analytics.is_internal_request(request),
    session_id=analytics.get_session_id(request),
)
```

**`get_distinct_id(request)`** — prefers `X-PostHog-Distinct-Id` header (frontend passes its PostHog ID to link events to the same person); falls back to a per-request UUID. Never uses client IP (shared NAT/VPN would merge unrelated users). Update this method when user auth is added.

**`get_session_id(request)`** — reads the `X-PostHog-Session-Id` header (forwarded by the frontend's `apiFetch` wrapper) and returns it, or `None` when absent (SSR / pre-session clients). Pass the result as `session_id=` to `capture_with_common_props` — the client injects it as PostHog's reserved `$session_id` property only when non-None, so session-scoped queries can link the server-side event to the client session.

**`is_internal_request(request)`** — returns `True` if `X-Internal-Request: true` header is present. Update this method to change the internal traffic detection strategy.

**`get_navigation_source(request)`** — reads the `X-Source` header and normalizes it against `KNOWN_NAVIGATION_SOURCES` (`"home" | "search" | "related" | "direct"`). Unknown or missing values become `"unknown"`. The frontend sets `X-Source` based on where the user clicked the link (or, for SSR, where the route arrived from). To add a new navigation source, extend the `KNOWN_NAVIGATION_SOURCES` frozenset in `src/analytics/client.py`.

**Disabled mode** — set `POSTHOG_API_KEY=` (empty) to silently disable all event tracking. All `capture_*` calls become no-ops. Useful for local dev without a PostHog project.

**Environment variables:** `POSTHOG_API_KEY`, `POSTHOG_HOST`, `APP_ENV`

**Testing analytics:**
- Unit tests for `AnalyticsClient`: `tests/analytics/test_analytics_client.py`
- Route-level analytics tests: `tests/server/articles/test_router.py`
- In tests, inject a `MagicMock(spec=AnalyticsClient)` via `app.dependency_overrides[get_analytics]` and assert on `capture_with_common_props` call args
- The `override_get_analytics` function must have NO parameters (or typed `Request`); an untyped `request` param causes FastAPI to treat it as a query param (422)

### Agent Architecture

The system uses a single specialized agent (`news_gatherer_agent`) that:
- Scrapes two specific Jamaica Gleaner sections: lead stories and news
- Identifies articles relevant to government accountability
- Returns structured JSON responses with relevance scoring
- Implements respectful crawling with 10-second delays

### Article Discovery System

The RSS-based article discovery system uses a multi-feed architecture that supports discovering articles from multiple RSS feeds simultaneously.

**Key Features:**
- **Multi-Feed Support**: Processes multiple RSS feeds in a single discovery operation
- **Per-Feed Sections**: Each feed can map to a different section (e.g., "lead-stories", "news")
- **Cross-Feed Deduplication**: Automatically deduplicates articles that appear in multiple feeds
- **Fail-Soft Error Handling**: If one feed fails, continues processing remaining feeds
- **Retry Logic**: Exponential backoff retry logic for network failures

**Components:**
- `GleanerRssFeedDiscoverer`: Main discoverer class that processes multiple feeds
- `RssFeedConfig`: Configuration dataclass for feed URL + section mapping
- `DiscoveredArticle`: Model for discovered article metadata (URL, title, section, dates)

**Usage Example:**
```python
from src.article_discovery.discoverers.gleaner_rss_discoverer import GleanerRssFeedDiscoverer
from src.article_discovery.models import RssFeedConfig

# Configure multiple feeds
feed_configs = [
    RssFeedConfig(
        url="https://jamaica-gleaner.com/feed/rss.xml",
        section="lead-stories"
    ),
    RssFeedConfig(
        url="https://jamaica-gleaner.com/feed/news.xml",
        section="news"
    )
]

# Discover articles from all feeds
discoverer = GleanerRssFeedDiscoverer(feed_configs=feed_configs)
articles = await discoverer.discover(news_source_id=1)
```

**Important Design Notes:**
- Feed URLs are passed as constructor parameters (not stored in database)
- Articles are deduplicated by URL across all feeds (first occurrence kept)
- Each feed is processed sequentially (not concurrently) to respect rate limits
- Discovery happens BEFORE extraction and classification
- No `article_id` at discovery stage (articles not yet stored)

**Testing:**
- Unit tests: `tests/article_discovery/discoverers/test_gleaner_rss_discoverer.py` (21 tests)
- Validation script: `scripts/validate_rss_discovery.py`

### Archive Discovery System

The archive discovery system uses date range-based discovery to find historical articles from newspaper archives (gleaner.newspaperarchive.com). It supports month-based discovery with parallel processing for improved performance.

**Key Features:**
- **Date Range Discovery**: Process articles by date ranges (days, months)
- **Month-Year Factory**: Convenient `for_month()` factory method for discovering entire months
- **Single Date Factory**: Convenient `for_date()` factory method for discovering specific dates (useful for retries)
- **Pagination Support**: Follows `<link rel="next">` tags to discover all pages per date
- **Respectful Crawling**: Configurable crawl delay between requests (default: 2 seconds)
- **Fail-Soft Error Handling**: Continues processing remaining dates if one fails
- **Retry Logic**: Exponential backoff for network failures
- **Parallel Processing**: External parallelization support via multiple workers

**Components:**
- `GleanerArchiveDiscoverer`: Main discoverer class for archive pages
- `deduplicate_discovered_articles()`: Helper function for cross-worker deduplication (in `utils.py`)

**Usage Examples:**

**Standard date range:**
```python
from datetime import datetime, timezone
from src.article_discovery.discoverers.gleaner_archive_discoverer import (
    GleanerArchiveDiscoverer,
)

# Discover last 7 days
discoverer = GleanerArchiveDiscoverer(
    end_date=datetime(2021, 11, 23, tzinfo=timezone.utc),
    days_back=7
)
articles = await discoverer.discover(news_source_id=1)
```

**Month-year discovery:**
```python
# Discover entire month (November 2021)
discoverer = GleanerArchiveDiscoverer.for_month(
    year=2021,
    month=11
)
articles = await discoverer.discover(news_source_id=1)
```

**Single date discovery:**
```python
# Discover specific date (November 15, 2021)
discoverer = GleanerArchiveDiscoverer.for_date(
    year=2021,
    month=11,
    day=15
)
articles = await discoverer.discover(news_source_id=1)
```

**Parallel multi-month discovery:**
```python
# Discover 3 months in parallel using 3 workers
import asyncio
from src.article_discovery.utils import deduplicate_discovered_articles

async def discover_month(year, month, news_source_id):
    discoverer = GleanerArchiveDiscoverer.for_month(
        year=year, month=month
    )
    return await discoverer.discover(news_source_id=news_source_id)

# Run 3 workers in parallel
results = await asyncio.gather(
    discover_month(2021, 9, 1),  # September
    discover_month(2021, 10, 1), # October
    discover_month(2021, 11, 1), # November
)

# Combine and deduplicate
all_articles = results[0] + results[1] + results[2]
unique_articles = deduplicate_discovered_articles(all_articles)
```

**Performance:**
- Sequential: ~30 seconds per date (20 pages × 2s delay + network)
- Month discovery: ~15-16 minutes for 30-day month (sequential)
- Parallel (3 workers): ~15-16 minutes for 3 months (3× speedup vs ~45-48 minutes sequential)
- Parallel (4 workers): ~45-48 minutes for 12 months (4× speedup vs ~3 hours sequential)

**Important Design Notes:**
- Factory methods validate year (1900-3000), month (1-12), and day (1-31) ranges
- `for_month()` factory creates discoverer for entire month (first to last day)
- `for_date()` factory creates discoverer for single date (useful for retrying failed dates)
- Date ranges are inclusive (first day to last day of month at midnight UTC)
- Each worker respects `crawl_delay` independently (no rate limit violations)
- Deduplication happens within each worker AND across workers
- No database interaction during discovery (only `news_source_id` passed through)
- Crawl delay is intentional for respectful crawling - parallelization is the only way to achieve speedup

**Scripts:**
- `scripts/parallel_archive_discovery.py` - CLI tool for parallel month discovery

**Usage:**
```bash
# Discover 3 months (Sep-Nov 2021) using 3 workers
uv run python scripts/parallel_archive_discovery.py \
    --year 2021 \
    --start-month 9 \
    --end-month 11 \
    --workers 3

# Discover with custom crawl delay
uv run python scripts/parallel_archive_discovery.py \
    --year 2021 \
    --start-month 1 \
    --end-month 12 \
    --workers 4 \
    --crawl-delay 1.0
```

**Testing:**
- Unit tests: `tests/article_discovery/discoverers/test_gleaner_archive_discoverer.py` (26 tests total)
  - 19 tests for core functionality
  - 7 tests for `for_month()` factory method
- Unit tests: `tests/article_discovery/test_utils.py` (4 tests for deduplication)
- Integration test: `tests/article_discovery/discoverers/test_gleaner_archive_discoverer_integration.py`

### Dependencies

**Agent System:**
- `google-adk>=1.8.0` - Google Agent Development Kit for LLM agent framework
- `litellm>=1.74.3` - LLM abstraction layer, configured to use o4-mini model
- `requests` - HTTP library for web scraping

**Database Layer:**
- `alembic>=1.17.1` - Database migration management
- `asyncpg>=0.30.0` - Async PostgreSQL driver
- `aiosql>=13.4` - SQL query management with raw SQL
- `greenlet>=3.2.3` - Required for SQLAlchemy async operations
- `pydantic>=2.11.7` - Data validation and serialization

**Environment:**
- Python 3.12+ required
- Docker & Docker Compose for local PostgreSQL database
- AWS CLI for LocalStack S3 bucket provisioning (`create-s3-buckets.sh`)

## Docker Setup

The project includes a production-ready Docker configuration.

**Key files:**
- `Dockerfile` — two-stage build: `builder` installs deps with uv into `.venv`; `runtime` copies the venv + source, runs as non-root user (UID 1001), starts `fastapi run` in production mode
- `docker-compose.yml` — two services: `postgres` (with healthcheck) and `app` (depends on postgres healthy)
- `.dockerignore` — excludes `.venv`, `.env*`, `tests/`, `scripts/`, `__pycache__`, etc.; `uv.lock` is intentionally included (needed by the builder stage)

**Environment variables** are never baked into the image. Locally, docker-compose reads them from `.env` via variable substitution. `DATABASE_URL` is constructed in `docker-compose.yml` to point to the `postgres` service name — this overrides any `DATABASE_URL` in `.env` when running via Docker Compose.

**Cold start workflow** (first run or after `docker-compose down -v`):
```bash
./scripts/infrastructure/compose-cold-start.sh
SEED_DB=true ./scripts/infrastructure/compose-cold-start.sh  # with seed
```

The script calls `start-db.sh`, `migrate.sh`, optionally `seed_db.sh`, then `docker-compose up -d app`. Normal restarts use `docker-compose up` directly (no migration step needed).

**Hot reload in development** — uncomment in `docker-compose.yml` under the `app` service:
```yaml
command: ["fastapi", "dev", "src/server/app.py", "--host", "0.0.0.0", "--port", "8000"]
volumes:
  - ./src:/app/src
```

## Development Commands

### Running the Application

**Via Google ADK web interface:**
```bash
adk web
```

**Via REST API (FastAPI development server):**
```bash
uv run fastapi dev src/server/app.py
```

- API available at `http://127.0.0.1:8000/`
- Interactive docs at `http://127.0.0.1:8000/docs`

**FastAPI production server:**
```bash
uv run fastapi run src/server/app.py
```

The FastAPI server lives in `src/server/app.py`. Add new routes there as the API grows.

### Database Management

**Helper Scripts** (all located in `scripts/infrastructure/`):
- `./scripts/infrastructure/start-db.sh` - Start PostgreSQL Docker container and wait for health check
- `./scripts/infrastructure/migrate.sh` - Run Alembic migrations to latest version
- `./scripts/infrastructure/create-migration.sh "message"` - Create new migration file with manual SQL DDL

**Common Operations:**
```bash
# Start database
./scripts/infrastructure/start-db.sh

# Run migrations
./scripts/infrastructure/migrate.sh

# Create new migration
./scripts/infrastructure/create-migration.sh "add classifications table"

# Connect to database
docker exec -it postgres psql -U user -d jaacountable_db

# View logs
docker-compose logs postgres

# Stop database
docker-compose down

# Stop and remove data (DESTRUCTIVE)
docker-compose down -v
```

**Database Credentials** (local development):
- User: `user`
- Password: `password`
- Database: `jaacountable_db`
- Port: `5432`
- Connection string: `postgresql+asyncpg://user:password@localhost:5432/jaacountable_db`

### Package Management

The project uses `uv` for dependency management:
- `uv.lock` - Lock file for reproducible builds
- `pyproject.toml` - Project configuration and dependencies

### Testing

The project uses pytest with testcontainers for isolated database testing. Each test session spins up a fresh PostgreSQL container and runs migrations automatically.

#### Test Categories

**Unit Tests:**
- Use mocks to simulate LLM responses
- No actual API calls made
- Fast execution
- Run by default during local development

**Integration Tests:**
- Make actual LLM API calls to validate end-to-end functionality
- Require `OPENAI_CLASSIFICATION_API_KEY` in `.env` file
- Slower execution and incur API costs
- Marked with `@pytest.mark.integration`
- Run during CI to ensure classifier works correctly

#### Running Tests

**Run all tests (unit + integration):**
```bash
uv run pytest tests/
```

**Run only unit tests (skip integration - no LLM API calls):**
```bash
uv run pytest tests/ -m "not integration"
```

**Run only integration tests (requires API key):**
```bash
uv run pytest tests/ -m integration
```

**Run tests with verbose output:**
```bash
uv run pytest tests/ -v
```

**Run tests with logging (shows database connection details):**
```bash
uv run pytest tests/ -v --log-cli-level=INFO
```

**Run a specific test file:**
```bash
uv run pytest tests/services/article_classification/test_corruption_classifier.py -v
```

**Run a specific test class:**
```bash
uv run pytest tests/article_persistence/repositories/test_article_repository.py::TestInsertArticleHappyPath -v
```

**Parallel Test Execution:**

The project includes pytest-xdist for running tests in parallel. Due to session-scoped async database fixtures, parallel execution works best across different test files:

```bash
# Run test files in parallel (each file runs in its own worker)
uv run pytest tests/ -n auto --dist loadfile

# Run only unit tests in parallel (skip integration)
uv run pytest tests/ -n auto --dist loadfile -m "not integration"
```

#### Test Features

- **Automatic PostgreSQL container lifecycle management**
- **Alembic migrations run automatically on test database**
- **Transaction rollback after each test for isolation**
- **BDD-style test format** (Given/When/Then comments)
- **Tests organized into classes by category** (HappyPath, ValidationErrors, DatabaseConstraints, EdgeCases)
- **pytest-xdist** available for parallel test execution across multiple test files
- **Integration test markers** registered in `pyproject.toml` for selective test execution

#### CI/CD

GitHub Actions CI runs **all tests** (including integration tests) using the `CLASSIFIER_AGENTS_CI` secret for the API key. This ensures the classifier works correctly with real LLM calls before merging.

**Test Infrastructure:**
- `tests/conftest.py` - Pytest fixtures for database testing
- `testcontainers` - Spins up isolated PostgreSQL containers per test session
- Automatic Alembic migrations on test database
- Transaction rollback after each test for isolation
- BDD-style test format (Given/When/Then comments)
- Tests organized into classes by category (e.g., `TestInsertArticleHappyPath`, `TestInsertArticleValidationErrors`)
- pytest-xdist for parallel test execution across test files

**Available Fixtures:**
- `postgres_container` - Session-scoped PostgreSQL container
- `test_database_url` - Connection URL from container
- `run_migrations` - Runs Alembic migrations on test database
- `db_pool` - asyncpg connection pool
- `db_connection` - Connection with automatic transaction rollback
- `redis_container` - Session-scoped Redis container (`redis:8-alpine`)
- `redis_client` - Function-scoped async Redis client; calls `FLUSHDB` after each test for isolation
- `redis_cache` - `RedisCacheBackend` wired to the test Redis container

Note: Tests within a single file run sequentially due to session-scoped async database fixtures and event loop constraints with pytest-asyncio.

The project also includes an evaluation set (`v1.evalset.json`) for testing agent performance.

## Important Implementation Details

### API Response Schemas
- **camelCase JSON**: All API response schemas must use camelCase field names in JSON output
- Use `alias_generator=to_camel` from `pydantic.alias_generators` in `ConfigDict` on every response schema
- Always include `populate_by_name=True` so internal Python code can still use snake_case attributes
- Query parameter schemas (e.g. `*Params`) are exempt — keep snake_case for URL query strings
- Example:
  ```python
  from pydantic.alias_generators import to_camel

  class MyResponseSchema(BaseModel):
      model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
      some_field: str  # serializes as "someField" in JSON
  ```

### Database Architecture
- **Raw SQL Approach**: Uses manual SQL DDL in Alembic migrations, NOT SQLAlchemy ORM models
- **Async Operations**: All database operations use asyncpg for async/await patterns
- **Connection Pooling**: `config/database.py` manages connection pools with proper lifecycle
- **Query Management**: Future queries will use aiosql with SQL stored in `.sql` files
- **Migration Strategy**: All migrations include both `upgrade()` and `downgrade()` functions
- **Environment Config**: Database URL loaded from `.env` file, managed by Alembic
- **UTC Timestamps**: All timestamp columns use `TIMESTAMPTZ` and Python code uses `datetime.now(timezone.utc)` for timezone-aware UTC datetimes

**Current Schema:**
- `articles` table: Stores scraped articles with deduplication via unique URL constraint
  - Indexes on `url` and `published_date` for performance
  - `fetched_at` (TIMESTAMPTZ) tracks when article was scraped
  - `published_date` (TIMESTAMPTZ) stores article publication date
  - `full_text` stores complete article content for classification

### Web Scraping Ethics
- The scraping tools implement a 10-second crawl delay to respect the target website
- Uses appropriate User-Agent headers
- Focuses only on two specific sections to minimize load

### Agent Prompt Engineering
- The agent is specifically instructed to look for government accountability topics
- Returns structured JSON with relevance scoring (1-10)
- Limited to maximum 20 articles per run to prevent overwhelming downstream systems
- Includes specific keywords and criteria for identifying relevant articles

### Data Flow
1. Agent calls `get_gleaner_lead_stories()` and `get_gleaner_news_section()`
2. Tools fetch HTML content with crawl delays
3. Agent processes content using LLM to identify relevant articles
4. Returns structured JSON with article metadata and relevance scores
5. (Future) Articles stored in PostgreSQL via asyncpg for persistence and deduplication

### Classification System Architecture

The project includes a modular classification system for AI-based article analysis. This system is separate from the initial news gathering agent and focuses on determining article relevance to specific accountability topics.

**Key Design Principles:**
- **Separation of Concerns**: Classification schemas (agent I/O) are separate from database models (persistence)
- **Classification Before Storage**: Articles are classified BEFORE database storage, so `article_id` is not available during classification
- **Type-Safe Classifier Identification**: Uses `ClassifierType` enum for type safety and JSON serialization
- **Comprehensive Validation**: Pydantic v2 models with extensive field validators
- **Extensible**: New classifiers can be added by extending the `ClassifierType` enum

**Classification Workflow:**
```
Extract → ClassificationInput → Classifier Agent → ClassificationResult → Store if relevant (article_id generated)
```

**Core Models (`src/services/article_classification/models.py`):**

1. **`ClassifierType` enum** (lines 7-24):
   - First enum in codebase using `(str, Enum)` pattern for JSON serialization
   - Type-safe classifier identification
   - Current values: `CORRUPTION`, `HURRICANE_RELIEF`
   - Add new classifiers by extending this enum

2. **`ClassificationInput` model** (lines 27-125):
   - Interface between extraction and classification layers
   - 5 fields: `url`, `title`, `section`, `full_text`, `published_date`
   - **Important**: No `article_id` field (classification happens before storage)
   - Validators for URL format, minimum text length (50 chars), timezone-aware datetimes
   - Constructed from `ExtractedArticleContent` + scraper context

3. **`ClassificationResult` model** (lines 128-199):
   - Structured output from classifier agents
   - 6 fields: `is_relevant`, `confidence`, `reasoning`, `key_entities`, `classifier_type`, `model_name`
   - `key_entities` defaults to `[]` (not `None`) for better ergonomics
   - `confidence` validated between 0.0 and 1.0
   - `model_name` included for traceability (tracks which LLM produced the result)
   - Validators for all fields with whitespace stripping and entity cleaning

**Test Coverage (`tests/services/article_classification/test_models.py`):**
- 62 total tests, all following BDD-style (Given/When/Then) format
- TestClassificationInputValidation: 25 tests
- TestClassifierTypeEnum: 5 tests
- TestClassificationResultValidation: 32 tests
- All tests use async pattern with pytest-asyncio
- Comprehensive validation testing, boundary conditions, edge cases (Unicode, very long text)

**Adding a New Classifier:**

1. **Update `ClassifierType` enum:**
   ```python
   class ClassifierType(str, Enum):
       CORRUPTION = "CORRUPTION"
       HURRICANE_RELIEF = "HURRICANE_RELIEF"
       NEW_CLASSIFIER = "NEW_CLASSIFIER"  # Add here
   ```

2. **Add enum test** in `TestClassifierTypeEnum`:
   ```python
   async def test_classifier_type_new_classifier_succeeds(self):
       result = ClassificationResult(
           is_relevant=True,
           confidence=0.85,
           reasoning="Test reasoning",
           classifier_type=ClassifierType.NEW_CLASSIFIER,
           model_name="gpt-4o-mini",
       )
       assert result.classifier_type == ClassifierType.NEW_CLASSIFIER
   ```

3. **Implement classifier agent** that accepts `ClassificationInput` and returns `ClassificationResult`

4. **Write comprehensive tests** for the new classifier agent

5. **Update database schema** if needed (create migration for classifier-specific columns)

**Important Notes:**
- Classification models use Pydantic v2 with `ConfigDict(from_attributes=True)`
- All validators use `@field_validator` decorator (Pydantic v2 pattern)
- Python 3.12+ type hints: `str | None` instead of `Optional[str]`
- `key_entities` uses list cleaning validator (strips whitespace, filters empty strings)
- All timestamp fields must be timezone-aware (`datetime.now(timezone.utc)`)
- First enum in codebase - established `(str, Enum)` pattern for JSON serialization

**Related Models:**
- `src/services/article_extractor/models.py` - `ExtractedArticleContent` (input to classification)
- `src/db/models/domain.py` - Database models for persistence (separate from classification schemas)