"""API schemas for the metrics domain."""
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class MetricsResponse(BaseModel):
    """Site-wide aggregate metrics for the hero section."""

    article_count: int
    entity_count: int

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
