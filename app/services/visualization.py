"""Visualization: AI chart suggestions (typed) and chart data payloads.

Charts are rendered on the frontend with Apache ECharts. The backend only
prepares the data (this module); the browser builds and draws the chart.
"""

import contextlib
import logging
from functools import cache

import numpy as np
import pandas as pd
from pydantic_ai import Agent

from .llm import get_model, run_with_backoff
from .schemas import Aggregation, ChartSpec, ChartSuggestions, SortOrder

logger = logging.getLogger("t2v.viz")

VIZ_SYSTEM_PROMPT = """
You are a data-visualization expert. Given the columns (with types) and a sample
of a query RESULT, suggest THREE short, natural-language chart ideas the user
could ask for, best first.

Each idea must be a self-contained instruction that can be executed by reshaping
THIS result alone (no new data is available) — reordering columns, grouping and
aggregating (sum/count/mean), sorting, or taking a top-N. Reference the result's
actual column names in plain language.

Write them the way a person would ask, for example:
- "Show total <measure> by <category>"
- "Show the trend of <measure> over <date column>"
- "Show the top 10 <category> by <measure>"
- "Show the share of <measure> across <category>"

Keep each idea under ~8 words. Return three distinct, genuinely useful ideas.
"""

CHART_SPEC_SYSTEM_PROMPT = """
You turn a natural-language chart request into a chart plan over an existing
query RESULT. You may ONLY use the columns listed in the user message — no new
data can be fetched.

Choose a chart_type from exactly these names:
"Bar Chart", "Line Chart", "Area Chart", "Pie Chart", "Scatter Plot", "Histogram".

Rules:
- x_column and y_column MUST be exact column names from the provided list.
- Pick the shape that fits the request and column types:
  - date/time x + numeric y      -> "Line Chart" or "Area Chart" (trends).
  - categorical x + numeric y     -> "Bar Chart"; "Pie Chart" only for a few
    parts-of-a-whole categories.
  - two numeric columns           -> "Scatter Plot".
  - one numeric column, distribution -> "Histogram" (set y_column to null).
- aggregation: use sum/count/mean/min/max when the request implies grouping the
  measure per x category (e.g. "total sales by region"); otherwise "none".
  Use "count" (y_column may be null) for "how many ... per ..." requests.
- sort/limit: set for "top/bottom N" or "highest/lowest" requests; else none.

SCOPE:
- If scope is "whole", plot the result as-is: aggregation "none", sort "none",
  no limit, x_column = first column, y_column = second column (or null for a
  single-column histogram).
- If scope is "result", you may subset/aggregate/sort as the request implies.
"""

@cache
def _get_viz_agent():
    return Agent(
        get_model(), output_type=ChartSuggestions, system_prompt=VIZ_SYSTEM_PROMPT
    )


@cache
def _get_spec_agent():
    return Agent(
        get_model(), output_type=ChartSpec, system_prompt=CHART_SPEC_SYSTEM_PROMPT
    )


def _describe_result(df):
    """Return (column->dtype, sample rows) used to prompt the LLM about a result."""
    return (
        {col: str(df[col].dtype) for col in df.columns},
        df.head(5).to_dict(orient="records"),
    )


def analyze_visualization(df):
    """Return up to three descriptive, executable chart ideas for the DataFrame."""
    column_info, sample_data = _describe_result(df)

    prompt = (
        f"Result columns and types: {column_info}\n"
        f"Sample rows: {sample_data}\n\n"
        f"Suggest three chart ideas for this result."
    )

    logger.info("Chart idea request | columns=%s", list(column_info.keys()))

    try:
        result = run_with_backoff(lambda: _get_viz_agent().run_sync(prompt))
        ideas = [idea.strip() for idea in result.output.ideas if idea.strip()]
        logger.info("LLM suggested chart ideas: %s", ideas)
        _log_usage(result)
    except Exception:
        logger.exception("Chart idea suggestion failed")
        return []  # let the caller render without suggestions

    return ideas


def generate_chart_spec(df, description, scope):
    """Turn a natural-language `description` into a validated ChartSpec over `df`.

    `scope` is "result" (may subset/aggregate) or "whole" (plot the result
    as-is). Returns a ChartSpec whose columns are guaranteed to exist in `df`.
    """
    column_info, sample_data = _describe_result(df)

    prompt = (
        f"Result columns and types: {column_info}\n"
        f"Sample rows: {sample_data}\n"
        f"Scope: {scope}\n\n"
        f"Chart request: {description}"
    )

    logger.info(
        "Chart spec request | scope=%s | columns=%s | request=%r",
        scope, list(column_info.keys()), description,
    )

    result = run_with_backoff(lambda: _get_spec_agent().run_sync(prompt))
    spec = result.output
    _log_usage(result)
    logger.info("LLM chart spec (raw): %s", spec)

    return _sanitize_spec(spec, df, scope)


def _sanitize_spec(spec, df, scope="result"):
    """Force the spec's columns to real ones, defaulting to the first two."""
    cols = list(df.columns)
    if spec.x_column not in cols:
        spec.x_column = cols[0]
    if spec.y_column is not None and spec.y_column not in cols:
        spec.y_column = cols[1] if len(cols) > 1 else None
    # x and y must differ, or shaped[[x, y]] yields duplicate columns.
    if spec.y_column == spec.x_column:
        others = [c for c in cols if c != spec.x_column]
        spec.y_column = others[0] if others else None
    # "whole" means plot the result as-is; don't let the model reshape it.
    if scope == "whole":
        spec.x_column = cols[0]
        spec.y_column = cols[1] if len(cols) > 1 else None
        spec.aggregation = Aggregation.none
        spec.sort = SortOrder.none
        spec.limit = None
    return spec


_AGG_FUNCS = {
    Aggregation.sum: "sum",
    Aggregation.mean: "mean",
    Aggregation.min: "min",
    Aggregation.max: "max",
}


def apply_chart_spec(df, spec):
    """Reshape `df` per `spec` (group/aggregate/sort/limit) and return a payload.

    Only columns already in `df` are used. Returns the same dict shape as
    `build_chart_payload`, including an {"error": ...} message on failure.
    """
    x, y = spec.x_column, spec.y_column

    # Histograms only need the single (numeric) column.
    if spec.chart_type == "Histogram":
        return build_chart_payload(df[[x]], "Histogram")

    if y is None and spec.aggregation != Aggregation.count:
        return {"error": "This chart needs a value column to plot."}

    shaped = df

    if spec.aggregation == Aggregation.count:
        # Avoid clashing with an existing column literally named "count".
        count_col = "count"
        while count_col in df.columns:
            count_col = f"_{count_col}"
        grouped = df.groupby(x, sort=False, dropna=False).size().reset_index(name=count_col)
        shaped, y = grouped, count_col
    elif spec.aggregation in _AGG_FUNCS:
        values = _to_numeric(df[y])
        grouped = (
            pd.DataFrame({x: df[x], y: values})
            .groupby(x, sort=False, dropna=False)[y]
            .agg(_AGG_FUNCS[spec.aggregation])
            .reset_index()
        )
        shaped = grouped

    if spec.sort != SortOrder.none and y in shaped.columns:
        shaped = shaped.sort_values(
            y, ascending=(spec.sort == SortOrder.asc),
            key=lambda s: _to_numeric(s),
        )

    if spec.limit and spec.limit > 0:
        shaped = shaped.head(spec.limit)

    payload = build_chart_payload(shaped[[x, y]], spec.chart_type)
    payload.setdefault("scope_columns", [x, y])
    return payload


def _log_usage(result):
    """Best-effort log of token usage from a PydanticAI run result."""
    with contextlib.suppress(Exception):
        logger.info("LLM usage: %s", result.usage)


def _to_numeric(series):
    """Coerce a Series to numbers, tolerating thousands separators and stray
    currency/percent symbols (e.g. "25,637.80" -> 25637.8) before falling back
    to NaN for anything genuinely non-numeric.
    """
    cleaned = series.astype(str).str.replace(r"[,\s₹$%]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def build_chart_payload(df, chart_type):
    """Prepare the JSON data ECharts needs to render `chart_type`.

    Returns a dict with either the chart data or an {"error": ...} message.
    """
    cols = list(df.columns)
    logger.info(
        "Chart data | type=%s | columns=%s | rows=%d", chart_type, cols, len(df)
    )

    if chart_type == "Histogram":
        series = _to_numeric(df[cols[0]]).dropna()
        if series.empty:
            return {"error": "The first column has no numeric values to plot."}
        bins = int(min(20, max(5, round(len(series) ** 0.5))))
        counts, edges = np.histogram(series, bins=bins)
        labels = [f"{edges[i]:.0f}-{edges[i + 1]:.0f}" for i in range(len(counts))]
        return {
            "chart_type": chart_type, "x_name": cols[0], "y_name": "count",
            "x": labels, "y": [int(v) for v in counts],
        }

    if len(cols) < 2:
        return {"error": "This chart needs at least two columns."}

    if chart_type == "Scatter Plot":
        x = _to_numeric(df[cols[0]])
        y = _to_numeric(df[cols[1]])
        mask = x.notna() & y.notna()
        return {
            "chart_type": chart_type, "x_name": cols[0], "y_name": cols[1],
            "points": [
                [float(a), float(b)] for a, b in zip(x[mask], y[mask], strict=False)
            ],
        }

    # Bar / Line / Area / Pie: first column = labels, second = numeric measure.
    x = df[cols[0]].astype(str).tolist()
    y = _to_numeric(df[cols[1]]).fillna(0)
    return {
        "chart_type": chart_type, "x_name": cols[0], "y_name": cols[1],
        "x": x, "y": [float(v) for v in y],
    }
