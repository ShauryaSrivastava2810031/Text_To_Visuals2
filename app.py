import os
import re
import time
import pandas as pd
import sqlalchemy
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_file, flash, Response
from flask_caching import Cache
from sqlalchemy import create_engine
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import inspect, text
import plotly.express as px
import pickle
import io
from io import StringIO

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Configure caching
cache = Cache(app, config={'CACHE_TYPE': 'simple', 'CACHE_DEFAULT_TIMEOUT': 600})  # Cache for 10 minutes

# Google API Key for LLM
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set. Please configure it in Azure.")

# PostgreSQL Database Setup
DB_URI = os.getenv("DATABASE_URL", "postgresql://postgres:comviDS2025@localhost:5432/User")
engine = create_engine(DB_URI)
db = SQLDatabase.from_uri(DB_URI)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0)

# Prompt for LLM to generate SQL queries
prompt = """
You are an AI assistant that converts natural language questions into SQL queries. 
Your task is to generate only the SQL query without explanations, comments, or extra text. 

- Database: PostgreSQL  
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
        return "TIMESTAMP"
    elif pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(series):
        return "DECIMAL"
    else:
        return "TEXT"

def safe_gemini_call(agent_executor, prompt, max_retries=5):
    """
    Calls the Gemini API safely, handling rate limits with exponential backoff.
    """
    retries = 0
    while retries < max_retries:
        try:
            return agent_executor.run(prompt).strip()
        except Exception as e:
            error_message = str(e)
            if "429 You exceeded your current quota" in error_message:
                match = re.search(r"retry_delay {\s*seconds: (\d+)", error_message)
                retry_time = int(match.group(1)) if match else (2 ** retries)  # Exponential backoff

                print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
                time.sleep(retry_time)
                retries += 1
            else:
                raise e  # Raise other errors normally

    raise RuntimeError("Max retries reached for Gemini API call. Please check your quota.")

def analyze_visualization(df):
    """
    This function initializes an AI agent that analyzes a dataset
    and suggests the top three most suitable visualizations.
    """
    sample_data = df.head(5).to_dict(orient="records")
    column_info = {col: str(df[col].dtype) for col in df.columns}

    # Identify if the dataset contains time-series data
    contains_time_column = any(dtype in ["datetime64[ns]", "DATE", "TIMESTAMP"] for dtype in column_info.values())

    # Define the agent prompt
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
    tools = [Tool(name="Visualization Suggestion", func=lambda x: x, description="Suggest best charts")]

    visualization_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )

    response = visualization_agent.run(visualization_prompt)

    # Validate AI response
    try:
        chart_suggestions = eval(response)

        # Ensure response is a valid list of exactly 3 visualizations
        if not isinstance(chart_suggestions, list) or len(chart_suggestions) != 3:
            raise ValueError("Invalid format")

        # Prioritize time-based charts if applicable
        if contains_time_column:
            chart_suggestions = sorted(chart_suggestions, key=lambda x: x in ["Line Chart", "Bar Chart", "Area Chart"], reverse=True)

    except:
        chart_suggestions = []  # No fallback list; AI must provide valid output

    return chart_suggestions


@app.route("/", methods=["GET", "POST"])
def upload_file():
    print("Route / triggered")
    global df, agent_executor, db
    if request.method == "POST":
        print("POST request received")

        if 'file' not in request.files:
            print("No file part found in request")
            return "No file part"

        file = request.files['file']
        if file.filename == '':
            print("No selected file")
            return "No selected file"
        if not allowed_file(file.filename):
            print("Invalid file type")
            return "Invalid file type"

        table_name = sanitize_column_name(os.path.splitext(file.filename)[0])
        print(f"Sanitized table name: {table_name}")

        # Initialize SQLAlchemy inspector
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()

        if table_name in existing_tables:
            print(f"Table '{table_name}' already exists. Skipping upload and using existing data.")

            db = SQLDatabase.from_uri(DB_URI)
            session['table_name'] = table_name
            print("Session updated with existing table")

            agent_executor = create_sql_agent(
                llm=llm,
                db=db,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True
            )
            print("Agent created using existing table")

            return redirect(url_for('query_interface'))

        filename = secure_filename(file.filename)
        filepath = os.path.join("uploads", filename)
        file.save(filepath)
        print(f"File saved to {filepath}")

        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        print(f"DataFrame loaded with shape: {df.shape}")

        df.columns = [sanitize_column_name(col) for col in df.columns]
        column_types = {col: detect_column_type(df[col]) for col in df.columns}
        print(f"Column types detected: {column_types}")

        # Drop all existing tables (per your current logic)
        with engine.connect() as conn:
            for existing_table in existing_tables:
                print(f"Dropping table {existing_table}")
                conn.execute(sqlalchemy.text(f"DROP TABLE IF EXISTS {existing_table}"))

        db = SQLDatabase.from_uri(DB_URI)

        with engine.connect() as conn:
            print("Creating new table")
            create_table_query = f"CREATE TABLE {table_name} ("
            create_table_query += ", ".join(f"{col} {dtype}" for col, dtype in column_types.items())
            create_table_query += ");"
            conn.execute(sqlalchemy.text(create_table_query))
            print("Table created successfully")

        df.to_sql(table_name, engine, if_exists="append", index=False)
        print("Data inserted into table")

        cache.clear()
        session['table_name'] = table_name
        print("Cache cleared and session updated")

        time.sleep(1)

        db = SQLDatabase.from_uri(DB_URI)
        print("SQLDatabase object created")

        agent_executor = create_sql_agent(
            llm=llm,
            db=db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        print("Agent created")

        print("Redirecting to query interface...")
        return redirect(url_for('query_interface'))

    return render_template("upload3.html")

@app.route("/home")
def home():
    global df, db, agent_executor
    df = None

    with engine.connect() as conn:
        inspector = inspect(engine)
        for table in inspector.get_table_names():
            conn.execute(sqlalchemy.text(f"DROP TABLE IF EXISTS {table}"))

    with engine.connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT").execute(text("VACUUM"))

    cache.clear()
    session.clear()

    return redirect(url_for('upload_file'))

@app.route("/query", methods=["GET", "POST"])
def query_interface():
    global df
    table_name = session.get('table_name', None)
    chart_suggestions = []

    if request.method == "POST":
        question = request.form["question"]

        contextual_prompt = f"You are working with a table named '{table_name}'. Answer the following query accordingly: {question}"

        cached_response = cache.get(contextual_prompt)
        if cached_response:
            sql_query, df_json, chart_options = cached_response
            df = pd.read_json(StringIO(df_json))
            chart_suggestions = chart_options
        else:
            try:
                sql_query = safe_gemini_call(agent_executor, prompt + contextual_prompt)
                conn = engine.connect()
                df = pd.read_sql_query(sql_query, conn)
                conn.close()

                if df is None or df.empty:
                    return render_template("index3.html", error="No data returned from query.")

                # Generate visualization suggestions only once (cached)
                chart_suggestions = analyze_visualization(df) or []
                cache.set(question, (sql_query, df.to_json(), chart_suggestions))
            except Exception as e:
                return render_template("index3.html", error=str(e))

        if len(df) > 100:
            # Don't render table for large data
            return render_template("index3.html", question=question, query=sql_query,
                                   table=None, chart_options=chart_suggestions,
                                   large_data=True)
        else:
            return render_template("index3.html", question=question, query=sql_query,
                                   table=df.to_html(classes="table table-stripped"),
                                   chart_options=chart_suggestions)

    return render_template("index3.html", chart_options=chart_suggestions)

@app.route("/generate_chart", methods=["POST"])
def generate_chart():
    global df  # ensure this is available at module level and has been initialized when dataset is uploaded

    chart_type = request.form.get("chart_type")
    print("Chart Selected:", chart_type)

    if "df" not in globals() or df is None or df.empty:
        return "<div class='alert alert-warning'>No data available for visualization.</div>"

    # Handle AI's non-graphical recommendations
    if chart_type.lower() == "numerical summary":
        summary_html = df.describe().to_html(classes="table table-bordered table-sm")
        return f"<h4>Numerical Summary</h4>{summary_html}"

    elif chart_type.lower() == "table view":
        table_html = df.to_html(classes="table table-bordered table-sm")
        return f"<h4>Table View</h4>{table_html}"

    # Handle graphical chart rendering
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

    # Convert figure to HTML for embedding
    graph_html = fig.to_html(full_html=False)

    return graph_html

@app.route("/download_csv")
def download_csv():
    table_name = session.get("table_name")
    if not table_name:
        return "No table loaded", 400

    df = pd.read_sql_table(table_name, con=engine)
    csv_data = df.to_csv(index=False)

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename={table_name}.csv"}
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
