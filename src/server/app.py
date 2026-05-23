"""FastAPI application entry point."""
import os
from contextlib import asynccontextmanager

import sentry_sdk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentry_sdk.integrations.logging import LoggingIntegration

from config.database import db_config
from config.log_config import configure_logging, logger
import redis.asyncio as aioredis

from src.analytics.client import analytics_client
from src.cache.in_memory import InMemoryCache
from src.cache.redis_cache import RedisCacheBackend
from src.server.articles.router import router as articles_router
from src.server.entities.router import router as entities_router
from src.server.health.router import router as health_router
from src.server.metrics.router import router as metrics_router
from src.server.middleware import CanonicalLogMiddleware

API_V1_PREFIX = "/api/v1"





@asynccontextmanager
async def lifespan(app: FastAPI):
    await db_config.create_pool()
    app.state.db_config = db_config
    app.state.analytics_client = analytics_client
    app.state.cache, _redis_client = await _init_cache()
    yield
    await db_config.close_pool()
    analytics_client.shutdown()
    if _redis_client:
        await _redis_client.aclose()

async def _init_cache() -> tuple[InMemoryCache | RedisCacheBackend, aioredis.Redis | None]:
    """Initialise cache backend from environment.

    Returns (cache_backend, redis_client_or_None).  The caller is responsible
    for closing the Redis client on shutdown when it is not None.
    """
    cache_url = os.getenv("CACHE_URL")
    ttl = int(os.getenv("CACHE_TTL_SECONDS", "300"))
    if cache_url:
        client = aioredis.from_url(cache_url)
        logger.info("Cache backend: Redis (url={} ttl={}s)", cache_url, ttl)
        return RedisCacheBackend(client=client, ttl_seconds=ttl), client
    logger.info("Cache backend: InMemoryCache (max_size=1,000 ttl={}s)", ttl)
    return InMemoryCache(ttl_seconds=ttl), None

def _init_sentry() -> None:
    """
    Initialize Sentry
    """
    dsn = os.getenv("SENTRY_DSN")
    if not dsn:
        logger.warning("Telemetry not initialized: SENTRY_DSN is not set")
        return

    sentry_sdk.init(
        dsn=dsn,
        environment=os.getenv("APP_ENV", "development"),
        send_default_pii=True,
        enable_logs=True,
        integrations=[
            # Disable stdlib log → Sentry Logs capture; LoguruIntegration handles it instead.
            # Without this, every stdlib log (e.g. uvicorn) is captured twice:
            # once via LoggingIntegration (auto.log.stdlib) and once via
            # InterceptHandler → loguru → LoguruIntegration (auto.log.loguru).
            LoggingIntegration(sentry_logs_level=None),
        ],
    )
    logger.info("Telemetry initialized")

# Begin App Startup

configure_logging()
_init_sentry()

app = FastAPI(lifespan=lifespan)

# Middlewares
app.add_middleware(CanonicalLogMiddleware)

#  Note on middleware ordering: FastAPI middlewares execute in reverse registration order (last registered = outermost).
#  CORS must run before CanonicalLogMiddleware so preflight OPTIONS requests are handled
#  and returned before hitting the canonical log layer.
#  Registering CORS after CanonicalLogMiddleware achieves this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://jaccountable.org",
        "https://staging.jaccountable.org",
        "http://localhost:3000",  # node dev server
        "http://localhost:4173",  # vite preview
    ],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Routers
app.include_router(health_router)
app.include_router(articles_router, prefix=API_V1_PREFIX)
app.include_router(entities_router, prefix=API_V1_PREFIX)
app.include_router(metrics_router, prefix=API_V1_PREFIX)
