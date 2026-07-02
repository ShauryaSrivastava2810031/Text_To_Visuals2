# Text to Visuals

A Flask app that turns natural-language questions into SQL (via LangChain +
Google Gemini), runs them against a PostgreSQL database, and renders Plotly
charts from the results.

## Project structure

```
run.py                     # entry point: creates the app, runs the dev server
app/
  __init__.py              # application factory (create_app)
  config.py                # configuration from environment variables
  extensions.py            # Flask extensions (cache)
  services/
    runtime.py             # in-memory holder for active DataFrame + SQL agent
    database.py            # engine, name/type inference, table creation
    llm.py                 # Gemini client, SQL agent, safe invocation
    visualization.py       # AI chart suggestions + Plotly rendering
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

All secrets live in `.env` (gitignored). See `.env.example` for the keys:
`GOOGLE_API_KEY`, `GEMINI_MODEL`, `DATABASE_URL`, `SECRET_KEY`.

## Known follow-ups (deferred implementation work)

- **LangChain 1.x migration (blocker):** the code uses the classic-agent API
  (`AgentType`, `initialize_agent`, `create_sql_agent` with `agent_type=`),
  which was removed in LangChain 1.0. The currently installed `langchain`
  is 1.3.x, so the LLM paths won't import until the agent code is migrated to
  the new API (or dependencies are pinned to the 0.x line).
- **Rotate `GOOGLE_API_KEY`:** the previous key was committed in source and
  must be treated as compromised.
- **Per-session state:** `services/runtime.py` holds the active DataFrame and
  agent as process globals (carried over from the original single-user design);
  make this per-session/thread-safe.
- **Read-only SQL execution:** LLM-generated SQL is executed directly; consider
  a read-only DB role and query validation.
