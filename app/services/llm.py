"""LLM access via PydanticAI.

Provider-agnostic: the active model is chosen by config (`LLM_PROVIDER` +
`LLM_MODEL`) and supports Google Gemini, OpenAI, and Anthropic.
"""

import contextlib
import logging
import re
import time
from functools import cache

from flask import current_app
from pydantic_ai import Agent

from .schemas import SqlQuery

logger = logging.getLogger("t2v.llm")

# System prompt for the text-to-SQL agent. Structured output guarantees we get
# only a SQL string back, so the old "return only the query" plumbing is gone.
SQL_SYSTEM_PROMPT = """
You convert a natural-language question into a single PostgreSQL query.

Rules:
- Read-only: output exactly ONE SELECT statement. Never INSERT, UPDATE, DELETE,
  DROP, ALTER, TRUNCATE, or return multiple statements.
- Use only the table and columns given in the user message. Never invent
  columns or reference other tables.
- Match the exact column names provided. If an identifier is a reserved word or
  contains uppercase or special characters, wrap it in double quotes.
- For text filters, match case-insensitively (use ILIKE, or
  LOWER(col) = LOWER('value')).
- When aggregating, include every non-aggregated selected column in GROUP BY and
  give aggregates clear aliases (e.g. SUM(amount) AS total_amount).
- For "top/bottom N" questions, use ORDER BY ... DESC/ASC with LIMIT N.
- Do not add LIMIT unless the question asks for a specific number of rows.
- Chart-friendly ordering: when the result is a grouped or time-based summary,
  put the category/label/date column FIRST and the numeric measure SECOND.
- If the question is ambiguous, choose the most reasonable interpretation.
- Return only the SQL: no comments, explanations, or markdown fences.
"""

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
    logger.info("Building LLM model | provider=%s | model=%s", provider, model_name)

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

    if provider == "openrouter":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openrouter import OpenRouterProvider

        return OpenAIChatModel(
            model_name,
            provider=OpenRouterProvider(api_key=_require_key("OPENROUTER_API_KEY")),
        )

    if provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        return AnthropicModel(
            model_name,
            provider=AnthropicProvider(api_key=_require_key("ANTHROPIC_API_KEY")),
        )

    raise RuntimeError(
        f"Unsupported LLM_PROVIDER {provider!r}. "
        "Use 'google', 'openai', 'anthropic', or 'openrouter'."
    )


@cache
def get_model():
    """Return the lazily-created, cached model for the configured provider."""
    return _build_model()


@cache
def get_sql_agent():
    """Return the lazily-created text-to-SQL agent."""
    return Agent(get_model(), output_type=SqlQuery, system_prompt=SQL_SYSTEM_PROMPT)


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
                logger.warning("Rate limit hit. Retrying in %ss...", wait)
                time.sleep(wait)
                retries += 1
                continue
            raise


def generate_sql(question, table_name, schema_text):
    """Generate a SQL query for `question` against the given table + schema."""
    user_prompt = (
        f"Table: {table_name}\n"
        f"Columns (name and type): {schema_text}\n\n"
        f"Question: {question}"
    )
    logger.info("SQL generation request | table=%s | question=%r", table_name, question)
    logger.debug("Schema passed to model: %s", schema_text)

    result = run_with_backoff(lambda: get_sql_agent().run_sync(user_prompt))
    sql = result.output.sql

    logger.info("Generated SQL: %s", sql)
    with contextlib.suppress(Exception):
        logger.info("LLM usage: %s", result.usage)
    return sql
