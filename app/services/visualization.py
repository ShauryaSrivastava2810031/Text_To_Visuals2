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
from .schemas import ChartSuggestions

logger = logging.getLogger("t2v.viz")

# Charts to prioritize when the dataset is time-based.
_TIME_PREFERRED = ["Line Chart", "Bar Chart", "Area Chart"]

VIZ_SYSTEM_PROMPT = """
You are a data-visualization expert. Given the columns (with types) and a sample
of a query RESULT, recommend the three most suitable chart types, best first.

The app renders a chart from the first two result columns: the first column is
the x-axis (category, label, or date) and the second is the y-axis (numeric
measure). Recommend charts that fit that shape and the column types.

Choose only from these exact names:
"Bar Chart", "Line Chart", "Area Chart", "Pie Chart", "Scatter Plot", "Histogram".

Guidance:
- Date/time column + numeric   -> "Line Chart" or "Area Chart" (trends over time).
- Categorical column + numeric -> "Bar Chart"; use "Pie Chart" only when there
  are few categories (<= 6) that represent parts of a whole.
- Two numeric columns          -> "Scatter Plot" (relationship between values).
- A single numeric column      -> "Histogram" (distribution).

Return three distinct charts that are genuinely appropriate for this result.
"""

@cache
def _get_viz_agent():
    return Agent(
        get_model(), output_type=ChartSuggestions, system_prompt=VIZ_SYSTEM_PROMPT
    )


def analyze_visualization(df):
    """Return up to three recommended chart names for the given DataFrame."""
    sample_data = df.head(5).to_dict(orient="records")
    column_info = {col: str(df[col].dtype) for col in df.columns}
    contains_time_column = any(
        dtype in ["datetime64[ns]", "DATE", "TIMESTAMP"]
        for dtype in column_info.values()
    )

    prompt = (
        f"Result columns and types: {column_info}\n"
        f"Sample rows: {sample_data}\n\n"
        f"Recommend the best three charts for this result."
    )

    logger.info(
        "Chart suggestion request | columns=%s | time_based=%s",
        list(column_info.keys()),
        contains_time_column,
    )

    try:
        result = run_with_backoff(lambda: _get_viz_agent().run_sync(prompt))
        chart_suggestions = [chart.value for chart in result.output.charts]
        logger.info("LLM suggested charts (raw): %s", chart_suggestions)
        _log_usage(result)
    except Exception:
        logger.exception("Chart suggestion failed")
        return []  # let the caller render without suggestions

    if contains_time_column:
        chart_suggestions = sorted(
            chart_suggestions, key=lambda x: x in _TIME_PREFERRED, reverse=True
        )
    logger.info("Final chart suggestions: %s", chart_suggestions)
    return chart_suggestions


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
