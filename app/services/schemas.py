"""Typed contracts for LLM outputs (validated by PydanticAI)."""

from enum import Enum

from pydantic import BaseModel, Field


class SqlQuery(BaseModel):
    """A single SQL query generated from a natural-language question."""

    sql: str = Field(
        description="A single valid PostgreSQL SELECT query, with no prose or markdown."
    )


class ChartType(str, Enum):
    """Charts the app knows how to render (see visualization.build_chart)."""

    bar = "Bar Chart"
    line = "Line Chart"
    area = "Area Chart"
    pie = "Pie Chart"
    scatter = "Scatter Plot"
    histogram = "Histogram"


class ChartSuggestions(BaseModel):
    """Exactly three recommended charts for a result set."""

    charts: list[ChartType] = Field(min_length=3, max_length=3)
