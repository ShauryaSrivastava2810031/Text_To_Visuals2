"""Visualization: AI chart suggestions and Plotly chart rendering."""

import ast
import re

import plotly.express as px
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool

from .llm import get_llm

# Charts that make sense to prioritize when the dataset is time-based.
_TIME_PREFERRED = ["Line Chart", "Bar Chart", "Area Chart"]


def _parse_chart_list(response):
    """Safely parse the LLM response into a list of exactly three chart names.

    Uses ast.literal_eval (never eval) and tolerates surrounding prose by
    extracting the first bracketed list found in the text.
    """
    candidate = response.strip()
    match = re.search(r"\[.*\]", candidate, re.DOTALL)
    if match:
        candidate = match.group(0)

    parsed = ast.literal_eval(candidate)
    if not isinstance(parsed, list) or len(parsed) != 3:
        raise ValueError("Expected a list of exactly three chart suggestions")
    return parsed


def analyze_visualization(df):
    """Ask the LLM for the top three visualizations for the given DataFrame."""
    sample_data = df.head(5).to_dict(orient="records")
    column_info = {col: str(df[col].dtype) for col in df.columns}

    contains_time_column = any(
        dtype in ["datetime64[ns]", "DATE", "TIMESTAMP"]
        for dtype in column_info.values()
    )

    visualization_prompt = f"""
    You are a data visualization expert.
    Analyze the following dataset structure and suggest the best visualization techniques.

    Column Information: {column_info}
    Sample Data: {sample_data}

    Rules:
    - Prefer "Line Chart" for time-based trends (e.g., sales over months).
    - Prefer "Bar Chart" for categorical vs numerical comparisons.
    - Prefer "Area Chart" for cumulative time-based data trends.
    - Prefer "Pie Chart" for categorical distributions.
    - Prefer "Scatter Plot" for numerical relationships.
    - Prefer "Histogram" for single-column numeric distributions.
    - Return exactly three suggestions in a Python list format.

    Now, suggest the best charts:
    """

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    tools = [
        Tool(
            name="Visualization Suggestion",
            func=lambda x: x,
            description="Suggest best charts",
        )
    ]

    visualization_agent = initialize_agent(
        tools=tools,
        llm=get_llm(),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
    )

    response = visualization_agent.run(visualization_prompt)

    try:
        chart_suggestions = _parse_chart_list(response)
        if contains_time_column:
            chart_suggestions = sorted(
                chart_suggestions,
                key=lambda x: x in _TIME_PREFERRED,
                reverse=True,
            )
    except (ValueError, SyntaxError):
        chart_suggestions = []  # AI must provide valid output; no hardcoded fallback

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
