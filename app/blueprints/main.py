"""Upload and reset routes."""

import os

import pandas as pd
from flask import (
    Blueprint,
    current_app,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from sqlalchemy import text
from werkzeug.utils import secure_filename

from ..extensions import cache
from ..services.database import (
    drop_all_tables,
    get_engine,
    list_tables,
    replace_table_with_dataframe,
    sanitize_column_name,
)
from ..services.llm import build_sql_agent, get_sql_database
from ..services.runtime import runtime

main_bp = Blueprint("main", __name__)


def allowed_file(filename):
    ext = filename.rsplit(".", 1)[-1].lower()
    return "." in filename and ext in current_app.config["ALLOWED_EXTENSIONS"]


@main_bp.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"

        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if not allowed_file(file.filename):
            return "Invalid file type"

        table_name = sanitize_column_name(os.path.splitext(file.filename)[0])

        # Reuse the table if it already exists.
        if table_name in list_tables():
            session["table_name"] = table_name
            runtime.sql_agent = build_sql_agent(get_sql_database())
            return redirect(url_for("query.query_interface"))

        # Save the uploaded file.
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Load into a DataFrame.
        if filename.endswith(".csv"):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        df.columns = [sanitize_column_name(col) for col in df.columns]

        # Replace existing data with the new table.
        replace_table_with_dataframe(table_name, df)

        cache.clear()
        session["table_name"] = table_name
        runtime.df = df
        runtime.sql_agent = build_sql_agent(get_sql_database())

        return redirect(url_for("query.query_interface"))

    return render_template("upload.html")


@main_bp.route("/home")
def home():
    """Reset everything: drop all tables, clear cache and session."""
    runtime.df = None
    runtime.sql_agent = None

    drop_all_tables()

    engine = get_engine()
    with engine.connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT").execute(text("VACUUM"))

    cache.clear()
    session.clear()

    return redirect(url_for("main.upload_file"))
