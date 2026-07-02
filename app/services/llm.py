"""LLM access via PydanticAI.

Provider-agnostic: the active model is chosen by config (`LLM_PROVIDER` +
`LLM_MODEL`) and supports Google Gemini, OpenAI, and Anthropic.
"""

import re
import time

from flask import current_app
from pydantic_ai import Agent

from .schemas import SqlQuery

# System prompt for the text-to-SQL agent. Structured output guarantees we get
# only a SQL string back, so the old "return only the query" plumbing is gone.
SQL_SYSTEM_PROMPT = """
You are an expert data analyst that converts natural language questions into SQL.

- Dialect: PostgreSQL.
- You are given the target table name and its columns in the user message.
- Produce a single, correct SELECT query that answers the question.
- Use the exact column and table names provided; do not invent columns.
- Do not add a LIMIT unless the question explicitly asks for one.
- Do not include comments, explanations, or markdown fences.
"""

_model = None
_sql_agent = None


def _require_key(config_key):
    key = current_app.config.get(config_key)
    if not key:
        raise RuntimeError(f"{config_key} is not set. Add it to your .env file.")
    return key


def _build_model():
    """Construct the configured provider's model. Imports are lazy so only the
    active provider's SDK needs to be importable."""
    provider = current_app.config["LLM_PROVIDER"].lower()
    model_name = current_app.config["LLM_MODEL"]

    if provider in ("google", "gemini", "google-gla"):
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        return GoogleModel(
            model_name, provider=GoogleProvider(api_key=_require_key("GOOGLE_API_KEY"))
        )

    if provider == "openai":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        return OpenAIChatModel(
            model_name, provider=OpenAIProvider(api_key=_require_key("OPENAI_API_KEY"))
        )

    if provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        return AnthropicModel(
            model_name,
            provider=AnthropicProvider(api_key=_require_key("ANTHROPIC_API_KEY")),
        )

    raise RuntimeError(
        f"Unsupported LLM_PROVIDER {provider!r}. Use 'google', 'openai', or 'anthropic'."
    )


def get_model():
    """Return the lazily-created, cached model for the configured provider."""
    global _model
    if _model is None:
        _model = _build_model()
    return _model


def get_sql_agent():
    """Return the lazily-created text-to-SQL agent."""
    global _sql_agent
    if _sql_agent is None:
        _sql_agent = Agent(
            get_model(), output_type=SqlQuery, system_prompt=SQL_SYSTEM_PROMPT
        )
    return _sql_agent


def run_with_backoff(fn, max_retries=5):
    """Run `fn`, retrying on rate-limit / quota errors with exponential backoff.

    Works across providers by matching common rate-limit signals in the error.
    """
    retries = 0
    while True:
        try:
            return fn()
        except Exception as e:
            message = str(e)
            rate_limited = "429" in message or "quota" in message.lower() or (
                "rate" in message.lower() and "limit" in message.lower()
            )
            if rate_limited and retries < max_retries:
                match = re.search(r"retry_delay {\s*seconds: (\d+)", message)
                wait = int(match.group(1)) if match else (2 ** retries)
                print(f"Rate limit hit. Retrying in {wait}s...")
                time.sleep(wait)
                retries += 1
                continue
            raise


def generate_sql(question, table_name, schema_text):
    """Generate a SQL query for `question` against the given table + schema."""
    user_prompt = (
        f"Table name: {table_name}\n"
        f"Columns: {schema_text}\n\n"
        f"Write a PostgreSQL query that answers: {question}"
    )
    result = run_with_backoff(lambda: get_sql_agent().run_sync(user_prompt))
    return result.output.sql
