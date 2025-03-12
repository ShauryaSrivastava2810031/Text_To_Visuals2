import os
import re
import time
import sqlite3
import pandas as pd
import sqlalchemy
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_caching import Cache
from sqlalchemy import create_engine
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from langchain.agents import AgentType
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import inspect
import plotly.express as px
import pickle

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Configure caching
cache = Cache(app, config={'CACHE_TYPE': 'simple', 'CACHE_DEFAULT_TIMEOUT': 300})  # Cache for 5 minutes

# Google API Key for LLM
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set. Please configure it in Azure.")

# SQLite Database Setup
DB_PATH = "sqlite:///user_data.db"
engine = create_engine(DB_PATH)

db = SQLDatabase.from_uri(DB_PATH)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=0)

# Prompt for LLM to generate SQL queries
prompt = """
You are an AI assistant that converts natural language questions into SQL queries. 
Your task is to generate only the SQL query without explanations, comments, or extra text. 

- Database: SQLite  
- Schema: Assume the database structure is already known to you.  
- Constraints: 
    1. Only return the SQL query. No extra text, explanations, or comments.  
    2. Ensure correctness with proper column names and table references.  
    3. Avoid assumptions if data is unavailable—return a valid, structured SQL query.
    4. Even if your final answer is a number, still return the SQL query as the final answer.
    5. Don't use unasked and unnecessary LIMIT keyword.

Now, generate the SQL query for:
"""

agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Allowed file types
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_column_name(name):
    return re.sub(r'\W+', '_', name).lower()

def detect_column_type(series):
    """
    Detects the appropriate SQL column type based on sample data in the Pandas Series.
    """
    sample_values = series.dropna().astype(str).head(20).tolist()

    date_patterns = [r'^\d{2}[-/]\d{2}[-/]\d{2,4}$', r'^\d{4}[-/]\d{2}[-/]\d{2}$', r'^\d{1,2}[a-z]{2} \w+ \d{4}$', r'^\w+ \d{1,2}, \d{4}$']
    time_patterns = [r'^\d{2}:\d{2}(:\d{2})?$']
    datetime_patterns = [r'^\d{2}/\d{2}/\d{4} \d{2}:\d{2}$', r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$']

    if all(any(re.match(pattern, val) for pattern in date_patterns) for val in sample_values if val):
        return "DATE"
    elif all(any(re.match(pattern, val) for pattern in time_patterns) for val in sample_values if val):
        return "TIME"
    elif all(any(re.match(pattern, val) for pattern in datetime_patterns) for val in sample_values if val):
        return "DATETIME"
    elif pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(series):
        return "DECIMAL"
    else:
        return "TEXT"

@app.route("/", methods=["GET", "POST"])
def upload_file():
    global df, agent_executor, db
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if file.filename == '':
            return "No selected file"
        if not allowed_file(file.filename):
            return "Invalid file type"

        table_name = sanitize_column_name(os.path.splitext(file.filename)[0])

        filename = secure_filename(file.filename)
        filepath = os.path.join("uploads", filename)
        file.save(filepath)

        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        df.columns = [sanitize_column_name(col) for col in df.columns]
        column_types = {col: detect_column_type(df[col]) for col in df.columns}

        inspector = inspect(engine)
        with engine.connect() as conn:
            for existing_table in inspector.get_table_names():
                conn.execute(sqlalchemy.text(f"DROP TABLE IF EXISTS {existing_table}"))

        db = SQLDatabase.from_uri(DB_PATH)

        with engine.connect() as conn:
            create_table_query = f"CREATE TABLE {table_name} ("
            create_table_query += ", ".join(f"{col} {dtype}" for col, dtype in column_types.items())
            create_table_query += ");"
            conn.execute(sqlalchemy.text(create_table_query))
            df.to_sql(table_name, engine, if_exists="append", index=False)

        cache.clear()
        session['table_name'] = table_name

        time.sleep(1)
        db = SQLDatabase.from_uri(DB_PATH)
        agent_executor = create_sql_agent(
            llm=llm,
            db=db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        return redirect(url_for('query_interface'))

    return render_template("upload.html")

@app.route("/home")
def home():
    global df, db, agent_executor
    df = None

    with engine.connect() as conn:
        inspector = inspect(engine)
        for table in inspector.get_table_names():
            conn.execute(sqlalchemy.text(f"DROP TABLE IF EXISTS {table}"))

    with engine.connect() as conn:
        conn.execute(sqlalchemy.text("VACUUM"))

    cache.clear()
    session.clear()

    return redirect(url_for('upload_file'))

@app.route("/query", methods=["GET", "POST"])
def query_interface():
    global df

    table_name = session.get('table_name', None)
    if request.method == "POST":
        question = request.form["question"]
        chart_type = request.form.get("chart_type", "bar chart")

        cached_response = cache.get(question)
        if cached_response:
            sql_query, df_json = cached_response
            df = pd.read_json(df_json)
        else:
            try:
                full_prompt = prompt + question
                response = agent_executor.run(full_prompt).strip()
                response = re.sub(r"sql|", "", response).strip()

                conn = sqlite3.connect("user_data.db")
                df = pd.read_sql_query(response, conn)
                conn.close()

                cache.set(question, (response, df.to_json()))
                cache.set("last_query_df", df)
                sql_query = response
            except Exception as e:
                return render_template("index.html", error=str(e))

        if df is None or df.empty:
            df_json = session.get('uploaded_data')
            if df_json:
                df = pd.read_json(df_json)

        fig = None

        if not df.empty and len(df.columns) > 1:
            if chart_type == "bar chart":
                fig = px.bar(df, x=df.columns[0], y=df.columns[1])
            elif chart_type == "pie chart":
                fig = px.pie(df, names=df.columns[0], values=df.columns[1])
            elif chart_type == "line chart":
                fig = px.line(df, x=df.columns[0], y=df.columns[1])
            elif chart_type == "scatter plot":
                fig = px.scatter(df, x=df.columns[0], y=df.columns[1])
            elif chart_type == "histogram":
                fig = px.histogram(df, x=df.columns[0])

        return render_template("index.html", question=question, query=sql_query,
                               table=df.to_html(classes="table table-stripped"),
                               chart=fig.to_html(full_html=False) if fig else None, chart_type=chart_type)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)