"""Visualization: AI chart suggestions (typed) and Plotly chart rendering."""

import logging

import plotly.express as px
from pydantic_ai import Agent

from .llm import get_model, run_with_backoff
from .schemas import ChartSuggestions

logger = logging.getLogger("t2v.viz")

# Charts to prioritize when the dataset is time-based.
_TIME_PREFERRED = ["Line Chart", "Bar Chart", "Area Chart"]

# App-aligned categorical palette + chart styling.
_PALETTE = ["#6366f1", "#a855f7", "#ec4899", "#f59e0b", "#10b981", "#06b6d4", "#ef4444"]
_ACCENT = "#6366f1"
_ACCENT_FILL = "rgba(99, 102, 241, 0.16)"
_FONT = "Inter, -apple-system, 'Segoe UI', Roboto, sans-serif"
_GRID = "#eef0f3"
_AXIS = "#e4e7eb"
_INK = "#16181d"

# Plotly modebar / interactivity config.
_PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}

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

_viz_agent = None


def _get_viz_agent():
    global _viz_agent
    if _viz_agent is None:
        _viz_agent = Agent(
            get_model(), output_type=ChartSuggestions, system_prompt=VIZ_SYSTEM_PROMPT
        )
    return _viz_agent


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
    try:
        logger.info("LLM usage: %s", result.usage())
    except Exception:
        pass


def _style_figure(fig, chart_type):
    """Apply the app's visual theme + interactivity to a Plotly figure."""
    fig.update_layout(
        height=470,
        template="plotly_white",
        colorway=_PALETTE,
        font=dict(family=_FONT, size=13, color=_INK),
        title=dict(
            font=dict(family=_FONT, size=18, color=_INK),
            x=0.012, xanchor="left", y=0.955, yanchor="top",
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=64, r=32, t=72, b=56),
        hoverlabel=dict(
            bgcolor="white", bordercolor=_AXIS,
            font=dict(family=_FONT, size=13, color=_INK),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(family=_FONT, size=12)),
        colorscale=dict(sequential=[[0, "#c7c9f6"], [1, _ACCENT]]),
    )
    fig.update_xaxes(
        showgrid=False, showline=True, linecolor=_AXIS, linewidth=1,
        ticks="outside", tickcolor=_AXIS, ticklen=6, title_font=dict(size=13),
        tickfont=dict(size=12),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=_GRID, zeroline=False, showline=False, ticks="",
        title_font=dict(size=13), tickfont=dict(size=12),
    )

    _spikes = dict(showspikes=True, spikecolor="#c7ccd4", spikethickness=1,
                   spikedash="dot", spikemode="across", spikesnap="cursor")
    _slider = dict(rangeslider=dict(visible=True, thickness=0.07, bgcolor="#f4f5f7"))

    if chart_type == "Line Chart":
        fig.update_traces(line=dict(width=3, color=_ACCENT),
                          mode="lines+markers", marker=dict(size=5, color=_ACCENT))
        fig.update_layout(hovermode="x unified")
        fig.update_xaxes(**_spikes, **_slider)
    elif chart_type == "Area Chart":
        fig.update_traces(line=dict(width=2.5, color=_ACCENT), mode="lines",
                          fill="tozeroy", fillcolor=_ACCENT_FILL)
        fig.update_layout(hovermode="x unified")
        fig.update_xaxes(**_spikes, **_slider)
    elif chart_type == "Scatter Plot":
        fig.update_traces(marker=dict(size=11, color=_ACCENT, opacity=0.8,
                                      line=dict(width=1.5, color="white")))
        fig.update_xaxes(**_spikes)
        fig.update_yaxes(showspikes=True, spikecolor="#c7ccd4", spikethickness=1,
                         spikedash="dot", spikemode="across")
    elif chart_type == "Bar Chart":
        fig.update_traces(marker=dict(color=_ACCENT, cornerradius=6, line_width=0))
        fig.update_layout(bargap=0.32)
    elif chart_type == "Histogram":
        fig.update_traces(marker=dict(color=_ACCENT, line=dict(color="white", width=1)),
                          opacity=0.92)
        fig.update_layout(bargap=0.05)
    elif chart_type == "Pie Chart":
        fig.update_traces(hole=0.58, textposition="outside", textinfo="percent+label",
                          marker=dict(line=dict(color="white", width=2)),
                          insidetextorientation="radial")
        fig.update_layout(showlegend=False)
    return fig


def build_chart(df, chart_type):
    """Render `chart_type` for `df`, returning an HTML fragment string."""
    normalized = chart_type.lower()
    logger.info("Render chart | type=%s | columns=%s | rows=%d",
                chart_type, list(df.columns), len(df))

    # Non-graphical recommendations
    if normalized == "numerical summary":
        summary_html = df.describe().to_html(classes="table table-bordered table-sm")
        return f"<h4>Numerical Summary</h4>{summary_html}"
    if normalized == "table view":
        table_html = df.to_html(classes="table table-bordered table-sm")
        return f"<h4>Table View</h4>{table_html}"

    fig = None
    if len(df.columns) > 1:
        x, y = df.columns[0], df.columns[1]
        try:
            if chart_type == "Bar Chart":
                fig = px.bar(df, x=x, y=y, title=f"{y} by {x}")
            elif chart_type == "Pie Chart":
                fig = px.pie(df, names=x, values=y, title=f"{y} by {x}")
            elif chart_type == "Line Chart":
                fig = px.line(df, x=x, y=y, markers=True, title=f"{y} over {x}")
            elif chart_type == "Area Chart":
                fig = px.area(df, x=x, y=y, markers=True, title=f"{y} over {x}")
            elif chart_type == "Scatter Plot":
                fig = px.scatter(df, x=x, y=y, title=f"{y} vs {x}")
            elif chart_type == "Histogram":
                fig = px.histogram(df, x=x, title=f"Distribution of {x}")
        except Exception as e:
            logger.exception("Chart build failed for %s", chart_type)
            return f"<div class='alert alert-danger'>Error generating chart: {str(e)}</div>"
    elif chart_type == "Histogram":
        fig = px.histogram(df, x=df.columns[0], title=f"Distribution of {df.columns[0]}")

    if fig is None:
        logger.warning("No renderer for chart_type=%s (columns=%s)",
                       chart_type, list(df.columns))
        return "<div class='alert alert-warning'>Invalid chart type selected.</div>"

    _style_figure(fig, chart_type)

    # Plotly.js is loaded once from the CDN in the page; emit only the chart div
    # + init script so each fragment stays small.
    return fig.to_html(full_html=False, include_plotlyjs=False, config=_PLOT_CONFIG)
