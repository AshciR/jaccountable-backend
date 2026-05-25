"""HTTP router for the articles domain."""
from typing import Annotated
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from src.analytics.client import AnalyticsClient
from src.server.articles.schemas import (
    ArticleSearchParams,
    ArticleSearchResponse,
    ArticleSearchResultSchema,
    RelatedArticlesResponse,
)
from src.cache.cache_interface import CacheBackend
from src.server.articles.service import ArticleSearchService
from src.server.dependencies import get_analytics, get_cache, get_db

router = APIRouter(prefix="/articles", tags=["articles"])


def _get_service(cache: Annotated[CacheBackend, Depends(get_cache)]) -> ArticleSearchService:
    return ArticleSearchService(cache=cache)


@router.get("", response_model=ArticleSearchResponse)
async def search_articles(
    request: Request,
    params: Annotated[ArticleSearchParams, Query()],
    conn: Annotated[asyncpg.Connection, Depends(get_db)],
    service: Annotated[ArticleSearchService, Depends(_get_service)],
    analytics: Annotated[AnalyticsClient, Depends(get_analytics)],
) -> ArticleSearchResponse:
    results, total = await service.search(conn, params)

    _capture_search_event(analytics, request, params, total)

    return ArticleSearchResponse.build(
        items=[ArticleSearchResultSchema.model_validate(r) for r in results],
        total=total,
        page=params.page,
        page_size=params.page_size,
    )


@router.get("/{public_id}/related", response_model=RelatedArticlesResponse)
async def get_related_articles(
    public_id: UUID,
    conn: Annotated[asyncpg.Connection, Depends(get_db)],
    service: Annotated[ArticleSearchService, Depends(_get_service)],
    limit: Annotated[int, Query(ge=1, le=20)] = 5,
    min_confidence: Annotated[float, Query(ge=0.0, le=1.0)] = 0.8,
) -> RelatedArticlesResponse:
    results = await service.get_related_articles(
        conn, public_id, limit=limit, min_confidence=min_confidence
    )
    return RelatedArticlesResponse(
        articles=[ArticleSearchResultSchema.model_validate(r) for r in results]
    )


@router.get("/{public_id}", response_model=ArticleSearchResultSchema)
async def get_article(
    request: Request,
    public_id: UUID,
    conn: Annotated[asyncpg.Connection, Depends(get_db)],
    service: Annotated[ArticleSearchService, Depends(_get_service)],
    analytics: Annotated[AnalyticsClient, Depends(get_analytics)],
) -> ArticleSearchResultSchema:
    result = await service.get_by_public_id(conn, public_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Article not found")

    _capture_detail_view_event(analytics, request, public_id, result)

    return ArticleSearchResultSchema.model_validate(result)


def _capture_detail_view_event(
    analytics: AnalyticsClient,
    request: Request,
    public_id: UUID,
    result,
) -> None:
    analytics.capture_with_common_props(
        distinct_id=analytics.get_distinct_id(request),
        event="article:detail_view",
        properties={
            "article_public_id": str(public_id),
            "article_title": result.title,
            "article_section": result.section,
            "navigation_source": analytics.get_navigation_source(request),
        },
        is_internal=analytics.is_internal_request(request),
        session_id=analytics.get_session_id(request),
    )


def _capture_search_event(
    analytics: AnalyticsClient,
    request: Request,
    params: ArticleSearchParams,
    total: int,
) -> None:
    distinct_id = analytics.get_distinct_id(request)
    is_internal = analytics.is_internal_request(request)
    session_id = analytics.get_session_id(request)
    # page == 1 means a fresh search or browse; page > 1 means the user clicked
    # Load More — the frontend reuses the same endpoint for both actions.
    if params.page == 1:
        # Skip empty-query loads (e.g. homepage default browse): they aren't real
        # searches and the frontend often fires them before PostHog has initialized,
        # so distinct_id / session_id come through wrong and pollute the search funnel.
        if not (params.q and params.q.strip()):
            return
        analytics.capture_with_common_props(
            distinct_id=distinct_id,
            event="search:query_submit",
            properties={"search_query": params.q, "results_count": total},
            is_internal=is_internal,
            session_id=session_id,
        )
    else:
        analytics.capture_with_common_props(
            distinct_id=distinct_id,
            event="search:load_more_click",
            properties={
                "search_query": params.q or "",
                "page": params.page,
                "current_results_count": (params.page - 1) * params.page_size,
            },
            is_internal=is_internal,
            session_id=session_id,
        )
