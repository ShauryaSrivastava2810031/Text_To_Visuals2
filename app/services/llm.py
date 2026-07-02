"""LLM access: Gemini client, SQL agent construction, and safe invocation."""

import re
import time

from flask import current_app
from langchain.agents import AgentType
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI

# Prompt prepended to user questions when generating SQL.
SQL_PROMPT = """
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

_llm = None


def get_llm():
    """Return a lazily-created, cached Gemini chat model."""
    global _llm
    if _llm is None:
        api_key = current_app.config["GOOGLE_API_KEY"]
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Add it to your .env file."
            )
        _llm = ChatGoogleGenerativeAI(
            model=current_app.config["GEMINI_MODEL"],
            google_api_key=api_key,
            temperature=0,
        )
    return _llm


def get_sql_database():
    """Build a LangChain SQLDatabase from the configured URI."""
    return SQLDatabase.from_uri(current_app.config["DATABASE_URL"])


def build_sql_agent(db=None):
    """Create a SQL agent bound to the given (or freshly loaded) database."""
    if db is None:
        db = get_sql_database()
    return create_sql_agent(
        llm=get_llm(),
        db=db,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )


def safe_agent_run(agent, prompt, max_retries=5):
    """Run the agent, handling Gemini rate limits with exponential backoff."""
    retries = 0
    while retries < max_retries:
        try:
            return agent.run(prompt).strip()
        except Exception as e:
            error_message = str(e)
            if "429 You exceeded your current quota" in error_message:
                match = re.search(r"retry_delay {\s*seconds: (\d+)", error_message)
                retry_time = int(match.group(1)) if match else (2 ** retries)
                print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
                time.sleep(retry_time)
                retries += 1
            else:
                raise e

    raise RuntimeError(
        "Max retries reached for Gemini API call. Please check your quota."
    )
