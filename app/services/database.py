"""Database helpers: engine access, name/type inference, table creation."""

import re
from functools import cache

import pandas as pd
import sqlalchemy
from flask import current_app
from sqlalchemy import create_engine, inspect


@cache
def get_engine():
    """Return a lazily-created, cached SQLAlchemy engine."""
    return create_engine(current_app.config["DATABASE_URL"])


def sanitize_column_name(name):
    """Normalize a column or table name to a safe snake_case identifier.

    Also prefixes an underscore when the result would start with a digit, since
    unquoted SQL identifiers may not begin with a number (e.g. "52w_h" -> "_52w_h").
    """
    identifier = re.sub(r"\W+", "_", name).lower()
    if identifier and identifier[0].isdigit():
        identifier = f"_{identifier}"
    return identifier


def detect_column_type(series):
    """Detect the appropriate SQL column type from sample values in a Series."""
    sample_values = series.dropna().astype(str).head(20).tolist()

    date_patterns = [
        r"^\d{2}[-/]\d{2}[-/]\d{2,4}$",
        r"^\d{4}[-/]\d{2}[-/]\d{2}$",
        r"^\d{1,2}[a-z]{2} \w+ \d{4}$",
        r"^\w+ \d{1,2}, \d{4}$",
    ]
    time_patterns = [r"^\d{2}:\d{2}(:\d{2})?$"]
    datetime_patterns = [
        r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}$",
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$",
    ]

    if all(any(re.match(p, v) for p in date_patterns) for v in sample_values if v):
        return "DATE"
    if all(any(re.match(p, v) for p in time_patterns) for v in sample_values if v):
        return "TIME"
    if all(any(re.match(p, v) for p in datetime_patterns) for v in sample_values if v):
        return "TIMESTAMP"
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series):
        return "DECIMAL"
    return "TEXT"


def list_tables():
    """Return the names of all tables in the connected database."""
    return inspect(get_engine()).get_table_names()


def get_table_schema(table_name):
    """Return a compact "col TYPE, col TYPE" description for the LLM prompt."""
    columns = inspect(get_engine()).get_columns(table_name)
    return ", ".join(f"{col['name']} {col['type']}" for col in columns)


def drop_all_tables():
    """Drop every table in the database (matches original reset behavior)."""
    engine = get_engine()
    with engine.connect() as conn:
        for table in inspect(engine).get_table_names():
            conn.execute(sqlalchemy.text(f"DROP TABLE IF EXISTS {table}"))
        conn.commit()


def replace_table_with_dataframe(table_name, df):
    """Drop all existing tables, then create `table_name` from `df` and load it.

    Preserves the original app's single-table replacement semantics.
    """
    engine = get_engine()
    column_types = {col: detect_column_type(df[col]) for col in df.columns}

    drop_all_tables()

    with engine.connect() as conn:
        columns_ddl = ", ".join(f"{col} {dtype}" for col, dtype in column_types.items())
        conn.execute(sqlalchemy.text(f"CREATE TABLE {table_name} ({columns_ddl});"))
        conn.commit()

    df.to_sql(table_name, engine, if_exists="append", index=False)
    return column_types
