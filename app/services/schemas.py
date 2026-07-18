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


class Aggregation(str, Enum):
    """How to collapse the y-column when grouping by the x-column."""

    none = "none"
    sum = "sum"
    count = "count"
    mean = "mean"
    min = "min"
    max = "max"


class SortOrder(str, Enum):
    """How to order rows before plotting (by the y/measure column)."""

    none = "none"
    asc = "asc"
    desc = "desc"


class ChartSuggestions(BaseModel):
    """Exactly three descriptive, executable chart ideas for a result set.

    Each phrase is a self-contained instruction the chart chatbox can run,
    e.g. "Show total sales by category" or "Show the top 10 rows by value".
    """

    ideas: list[str] = Field(min_length=3, max_length=3)


class ChartSpec(BaseModel):
    """A chart plan derived from a natural-language description of the result.

    Reshape-only: `x_column`/`y_column` must be columns already present in the
    result; no new data is fetched. The backend validates the columns and
    applies the aggregation/sort/limit with pandas before rendering.
    """

    chart_type: ChartType
    x_column: str = Field(description="Result column for the x-axis (category/label/date).")
    y_column: str | None = Field(
        default=None,
        description="Result column for the y-axis (numeric measure). "
        "May be null for Histogram or a count of the x-column.",
    )
    aggregation: Aggregation = Aggregation.none
    sort: SortOrder = SortOrder.none
    limit: int | None = Field(
        default=None, description="Keep only the first N rows after sorting, if requested."
    )
