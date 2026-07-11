"""Query interface and CSV download routes."""

from io import StringIO

import pandas as pd
from flask import Blueprint, Response, current_app, render_template, request, session

from ..extensions import cache
from ..services.database import get_engine, get_table_schema
from ..services.llm import generate_sql
from ..services.runtime import runtime
from ..services.visualization import analyze_visualization

query_bp = Blueprint("query", __name__)


@query_bp.route("/query", methods=["GET", "POST"])
def query_interface():
    """Answer a question: generate SQL, run it, and render the results."""
    table_name = session.get("table_name")
    chart_suggestions = []

    if request.method == "POST":
        question = request.form["question"]
        cache_key = f"{table_name}:{question}"

        cached_response = cache.get(cache_key)
        if cached_response:
            sql_query, df_json, chart_suggestions = cached_response
            df = pd.read_json(StringIO(df_json))
        else:
            try:
                schema_text = get_table_schema(table_name)
                sql_query = generate_sql(question, table_name, schema_text)
                with get_engine().connect() as conn:
                    df = pd.read_sql_query(sql_query, conn)

                if df is None or df.empty:
                    return render_template(
                        "index.html", error="No data returned from query."
                    )

                chart_suggestions = analyze_visualization(df) or []
                # Cache under the same key used for lookup above.
                cache.set(cache_key, (sql_query, df.to_json(), chart_suggestions))
            except Exception as e:
                return render_template("index.html", error=str(e))

        # Share the result with the chart route, then render.
        runtime.df = df

        if len(df) > current_app.config["MAX_TABLE_ROWS"]:
            return render_template(
                "index.html",
                question=question,
                query=sql_query,
                table=None,
                chart_options=chart_suggestions,
                large_data=True,
            )
        return render_template(
            "index.html",
            question=question,
            query=sql_query,
            table=df.to_html(classes="table"),
            chart_options=chart_suggestions,
        )

    return render_template("index.html", chart_options=chart_suggestions)


@query_bp.route("/download_csv")
def download_csv():
    """Stream the active table as a downloadable CSV."""
    table_name = session.get("table_name")
    if not table_name:
        return "No table loaded", 400

    df = pd.read_sql_table(str(table_name), con=get_engine())
    csv_data = df.to_csv(index=False)

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename={table_name}.csv"},
    )
