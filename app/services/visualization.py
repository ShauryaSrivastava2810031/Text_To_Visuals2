"""Visualization: AI chart suggestions (typed) and Plotly chart rendering."""

import plotly.express as px
from pydantic_ai import Agent

from .llm import get_model, run_with_backoff
from .schemas import ChartSuggestions

# Charts to prioritize when the dataset is time-based.
_TIME_PREFERRED = ["Line Chart", "Bar Chart", "Area Chart"]

VIZ_SYSTEM_PROMPT = """
You are a data visualization expert. Given a dataset's columns and a small
sample, recommend exactly three suitable chart types. Guidelines:
- Prefer "Line Chart" for time-based trends.
- Prefer "Bar Chart" for categorical vs numerical comparisons.
- Prefer "Area Chart" for cumulative time-based trends.
- Prefer "Pie Chart" for categorical distributions.
- Prefer "Scatter Plot" for numerical relationships.
- Prefer "Histogram" for single-column numeric distributions.
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
        f"Column Information: {column_info}\n"
        f"Sample Data: {sample_data}\n\n"
        f"Suggest the best three charts."
    )

    try:
        result = run_with_backoff(lambda: _get_viz_agent().run_sync(prompt))
        chart_suggestions = [chart.value for chart in result.output.charts]
    except Exception:
        return []  # let the caller render without suggestions

    if contains_time_column:
        chart_suggestions = sorted(
            chart_suggestions, key=lambda x: x in _TIME_PREFERRED, reverse=True
        )
    return chart_suggestions


def build_chart(df, chart_type):
    """Render `chart_type` for `df`, returning an HTML fragment string."""
    normalized = chart_type.lower()

    # Non-graphical recommendations
    if normalized == "numerical summary":
        summary_html = df.describe().to_html(classes="table table-bordered table-sm")
        return f"<h4>Numerical Summary</h4>{summary_html}"
    if normalized == "table view":
        table_html = df.to_html(classes="table table-bordered table-sm")
        return f"<h4>Table View</h4>{table_html}"

    fig = None
    if len(df.columns) > 1:
        try:
            if chart_type == "Bar Chart":
                fig = px.bar(df, x=df.columns[0], y=df.columns[1])
            elif chart_type == "Pie Chart":
                fig = px.pie(df, names=df.columns[0], values=df.columns[1])
            elif chart_type == "Line Chart":
                fig = px.line(df, x=df.columns[0], y=df.columns[1])
            elif chart_type == "Scatter Plot":
                fig = px.scatter(df, x=df.columns[0], y=df.columns[1])
            elif chart_type == "Histogram":
                fig = px.histogram(df, x=df.columns[0])
        except Exception as e:
            return f"<div class='alert alert-danger'>Error generating chart: {str(e)}</div>"

    if fig is None:
        return "<div class='alert alert-warning'>Invalid chart type selected.</div>"

    return fig.to_html(full_html=False)
