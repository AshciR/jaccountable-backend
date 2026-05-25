"""API schemas for the articles domain."""
import math
from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel


class ArticleSearchParams(BaseModel):
    """Query parameters for the article search endpoint."""

    q: str | None = None
    from_date: datetime | None = None
    to_date: datetime | None = None
    include_full_text: bool = False
    page: int = Field(1, ge=1)
    page_size: int = Field(20, ge=1, le=100)
    sort: Literal["relevance", "published_date"] = "relevance"
    order: Literal["asc", "desc"] = "desc"
    min_confidence: float = Field(0.8, ge=0.0, le=1.0)


class SearchClassificationSchema(BaseModel):
    """Classification summary embedded in a search result."""

    classifier_type: str
    confidence_score: float
    reasoning: str | None = None

    model_config = ConfigDict(from_attributes=True, alias_generator=to_camel, populate_by_name=True)


class NewsSource(str, Enum):
    JAMAICA_GLEANER = "JAMAICA_GLEANER"
    JAMAICA_OBSERVER = "JAMAICA_OBSERVER"


_NEWS_SOURCE_MAP: dict[int, NewsSource] = {
    1: NewsSource.JAMAICA_GLEANER,
    2: NewsSource.JAMAICA_OBSERVER,
}


class ArticleSearchResultSchema(BaseModel):
    """Single article result returned by the search endpoint."""

    id: UUID = Field(validation_alias="public_id")
    url: str
    title: str
    section: str
    published_date: datetime | None
    news_source: NewsSource = Field(validation_alias="news_source_id")
    snippet: str | None
    entities: list[str]
    classifications: list[SearchClassificationSchema]
    full_text: str | None = None

    model_config = ConfigDict(from_attributes=True, alias_generator=to_camel, populate_by_name=True)

    @field_validator("news_source", mode="before")
    @classmethod
    def map_news_source(cls, v: int | str | NewsSource) -> NewsSource:
        if isinstance(v, int):
            mapped = _NEWS_SOURCE_MAP.get(v)
            if mapped is None:
                raise ValueError(f"Unknown news_source_id: {v}")
            return mapped
        return NewsSource(v)

class RelatedArticlesResponse(BaseModel):
    """Response wrapper for related articles."""

    articles: list[ArticleSearchResultSchema]

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ArticleSearchResponse(BaseModel):
    """Paginated article search response."""

    items: list[ArticleSearchResultSchema]
    total: int
    page: int
    page_size: int
    pages: int

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    @classmethod
    def build(
        cls,
        items: list[ArticleSearchResultSchema],
        total: int,
        page: int,
        page_size: int,
    ) -> "ArticleSearchResponse":
        pages = math.ceil(total / page_size) if total else 0
        return cls(items=items, total=total, page=page, page_size=page_size, pages=pages)
