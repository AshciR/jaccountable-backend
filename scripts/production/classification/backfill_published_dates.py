"""
Backfill missing published dates for Jamaica Gleaner articles.

The Gleaner extractor previously only recognised JSON-LD typed "Article",
but the Gleaner now emits "NewsArticle", so articles were stored with
published_date = NULL. Gleaner article URLs embed the publish date
(/article/{section}/{YYYYMMDD}/{slug}), so this script fills the gap from
the URL without any network calls.

Dates are stored at midnight UTC, matching the sitemap/archive discoverers.
Only rows that are still NULL are updated, so the script is safe to re-run.

Usage:
    # Preview changes (transaction is rolled back)
    uv run python scripts/production/classification/backfill_published_dates.py --dry-run

    # Apply
    uv run python scripts/production/classification/backfill_published_dates.py

Environment Variables:
    DATABASE_URL    PostgreSQL connection string
    LOG_JSON        Enable JSON logging (default: false)
"""
import argparse
import asyncio
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import asyncpg
from dotenv import load_dotenv
from loguru import logger

# Add project root to path
# Script is at scripts/production/classification/backfill_published_dates.py
# So we need to go up 3 levels to reach project root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from config.database import db_config
from config.log_config import configure_logging

GLEANER_NEWS_SOURCE_NAME = "Jamaica Gleaner"

# Same pattern as JamaicaGleanerSitemapDiscoverer: /article/{section}/{YYYYMMDD}/{slug}
_GLEANER_URL_DATE_RE = re.compile(r"/article/[^/]+/(\d{4})(\d{2})(\d{2})/[^/]+")


@dataclass
class BackfillSummary:
    """Outcome of a backfill run."""

    candidates: int = 0
    updated: int = 0
    unparseable_urls: list[str] = field(default_factory=list)
    updates: list[dict[str, str]] = field(default_factory=list)


def parse_gleaner_url_date(url: str) -> datetime | None:
    """
    Extract the publish date embedded in a Jamaica Gleaner article URL.

    Args:
        url: Article URL.

    Returns:
        Timezone-aware datetime at midnight UTC, or None if the URL has no
        valid date component.
    """
    match = _GLEANER_URL_DATE_RE.search(url)
    if not match:
        return None
    year, month, day = (int(g) for g in match.groups())
    try:
        return datetime(year, month, day, tzinfo=timezone.utc)
    except ValueError:
        return None


async def backfill_published_dates(
    conn: asyncpg.Connection,
    dry_run: bool = False,
    limit: int | None = None,
) -> BackfillSummary:
    """
    Fill NULL published_date values for Gleaner articles from their URLs.

    Args:
        conn: Database connection (caller manages lifecycle).
        dry_run: If True, run the updates inside a transaction that is
            always rolled back.
        limit: Max number of candidate rows to process (None = all).

    Returns:
        BackfillSummary with counts, applied updates and unparseable URLs.
    """
    rows = await conn.fetch(
        """
        SELECT a.id, a.url
        FROM articles a
        JOIN news_sources ns ON ns.id = a.news_source_id
        WHERE ns.name = $1
          AND a.published_date IS NULL
        ORDER BY a.id
        LIMIT $2
        """,
        GLEANER_NEWS_SOURCE_NAME,
        limit,
    )

    summary = BackfillSummary(candidates=len(rows))
    updates: list[tuple[int, datetime]] = []
    for row in rows:
        published_date = parse_gleaner_url_date(row["url"])
        if published_date is None:
            summary.unparseable_urls.append(row["url"])
            continue
        updates.append((row["id"], published_date))
        summary.updates.append({"url": row["url"], "published_date": published_date.isoformat()})

    tx = conn.transaction()
    await tx.start()
    try:
        for article_id, published_date in updates:
            status = await conn.execute(
                "UPDATE articles SET published_date = $2 WHERE id = $1 AND published_date IS NULL",
                article_id,
                published_date,
            )
            summary.updated += int(status.split()[-1])
    except Exception:
        await tx.rollback()
        raise
    if dry_run:
        await tx.rollback()
    else:
        await tx.commit()

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Backfill NULL published dates for Jamaica Gleaner articles from their URLs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview changes (rolled back)
  uv run python scripts/production/classification/backfill_published_dates.py --dry-run

  # Apply to the first 100 candidates only
  uv run python scripts/production/classification/backfill_published_dates.py --limit 100
        """,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and apply updates inside a transaction that is rolled back",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of articles to process (default: all)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scripts/production/classification/output"),
        help="Output directory for results (default: scripts/production/classification/output)",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    return args


async def main() -> int:
    """
    Main entry point with full lifecycle management.

    Returns:
        Exit code (0=success, 1=error)
    """
    load_dotenv()
    args = parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    file_stem = f"gleaner_backfill_published_dates_{timestamp}"

    logs_dir = args.output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{file_stem}.log"
    configure_logging(enable_file_logging=True, log_file_path=str(log_file))

    logger.info("=" * 80)
    logger.info("PUBLISHED DATE BACKFILL STARTED")
    logger.info(f"Dry-run mode: {args.dry_run}")
    logger.info(f"Limit: {args.limit or 'none'}")
    logger.info("=" * 80)

    exit_code = 0
    try:
        await db_config.create_pool(command_timeout=60.0)
        async with db_config.connection() as conn:
            summary = await backfill_published_dates(conn, dry_run=args.dry_run, limit=args.limit)

        results_dir = args.output_dir / "backfill_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"{file_stem}.json"
        results_file.write_text(
            json.dumps({"dry_run": args.dry_run, "limit": args.limit, **asdict(summary)}, indent=2)
        )

        verb = "Would update" if args.dry_run else "Updated"
        logger.info(f"Candidates (Gleaner, published_date IS NULL): {summary.candidates}")
        logger.info(f"{verb}: {summary.updated}")
        logger.info(f"No date in URL (left NULL): {len(summary.unparseable_urls)}")
        for url in summary.unparseable_urls[:20]:
            logger.info(f"  {url}")
        logger.info(f"Results: {results_file}")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        exit_code = 1
    finally:
        await db_config.close_pool()

    logger.remove()
    return exit_code


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
