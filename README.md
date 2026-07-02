# Text to Visuals

A Flask app that turns natural-language questions into SQL (via PydanticAI,
provider-agnostic across Google Gemini, OpenAI, and Anthropic), runs them
against a PostgreSQL database, and renders Plotly charts from the results.

## Project structure

```
run.py                     # entry point: creates the app, runs the dev server
app/
  __init__.py              # application factory (create_app)
  config.py                # configuration from environment variables
  extensions.py            # Flask extensions (cache)
  services/
    schemas.py             # typed LLM output models (SqlQuery, ChartSuggestions)
    runtime.py             # in-memory holder for the active DataFrame
    database.py            # engine, name/type inference, table + schema helpers
    llm.py                 # provider-agnostic model + text-to-SQL agent
    visualization.py       # AI chart suggestions (typed) + Plotly rendering
  blueprints/
    main.py                # "/" upload, "/home" reset
    query.py               # "/query" NL->SQL, "/download_csv"
    chart.py               # "/generate_chart" (AJAX)
  templates/
    upload.html
    index.html
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

copy .env.example .env            # then fill in real values
python run.py                     # serves on http://localhost:8000
```

Deployment entry point (e.g. Azure/gunicorn): `run:app`
(`gunicorn run:app`).

## Configuration

All secrets live in `.env` (gitignored). See `.env.example` for the keys.

The LLM is provider-agnostic via [PydanticAI](https://ai.pydantic.dev). Pick a
provider and model in `.env`:

| `LLM_PROVIDER` | Example `LLM_MODEL` | Required key |
|----------------|---------------------|--------------|
| `google`       | `gemini-2.0-flash`  | `GOOGLE_API_KEY` |
| `openai`       | `gpt-4o`            | `OPENAI_API_KEY` |
| `anthropic`    | `claude-sonnet-4-5` | `ANTHROPIC_API_KEY` |

Only the active provider's key is required.

## Known follow-ups (deferred implementation work)

- **Rotate `GOOGLE_API_KEY`:** the previous key was committed in source and
  must be treated as compromised.
- **SQL self-correction:** generation is a single structured call; a bad query
  surfaces as an error. Optionally give the SQL agent a `run_sql` tool so it can
  validate/retry before returning.
- **Per-session state:** `services/runtime.py` holds the active DataFrame as a
  process global (carried over from the original single-user design); make this
  per-session/thread-safe.
- **Read-only SQL execution:** LLM-generated SQL is executed directly; consider
  a read-only DB role and query validation.
