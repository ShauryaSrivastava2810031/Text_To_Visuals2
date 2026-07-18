# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Flask app that turns natural-language questions into SQL (via **PydanticAI**,
provider-agnostic across Google Gemini, OpenAI, Anthropic, and OpenRouter), runs
them against **PostgreSQL**, and returns data payloads that the browser renders as
interactive **Apache ECharts**. The backend never draws a chart — it only prepares
JSON; the frontend (`app/templates/index.html`) builds and draws it.

## Commands

Setup and running go through the `justfile` (see https://just.systems). On Windows
recipes run under PowerShell; the venv is created with the `py` launcher (`python`
is not on PATH) and everything else uses `.venv/Scripts/python.exe`.

- `just setup` — one-time: create venv, install deps, copy `.env.example` → `.env`, run migrations
- `just dev` — dev server on http://localhost:8000 (auto-reload only if `FLASK_DEBUG=1` in `.env`)
- `just serve` — gunicorn (Linux/macOS only)
- `just update` — reinstall deps after `requirements.txt` changes
- `just clean` — remove the venv

Manual run (no `just`): `.venv\Scripts\python.exe run.py`. Deployment entry point is
`run:app` (e.g. `gunicorn run:app`).

There is **no test suite, linter config, or migration tool** wired in yet (`just db`
is a documented no-op until one is added). Do not assume `pytest`/`ruff` exist.

## Configuration

All config comes from environment variables via `app/config.py`, loaded from a
gitignored `.env` (see `.env.example`). The LLM is selected at runtime by
`LLM_PROVIDER` + `LLM_MODEL`; only the active provider's API key is required.
Supported providers: `google`, `openai`, `anthropic`, `openrouter`.

## Architecture

Standard Flask app-factory layout. `create_app()` (`app/__init__.py`) wires config,
the `t2v.*` console loggers, the `flask-caching` SimpleCache, and three blueprints.

**Request flow:**

1. **`blueprints/main.py`** (`/`, `/home`) — CSV/XLSX upload. Column and table names
   are run through `sanitize_column_name`; column SQL types are inferred by sampling
   values (`detect_column_type`). Uploading **replaces all data**:
   `replace_table_with_dataframe` drops every existing table first (single-table,
   single-dataset design). `/home` resets everything (drop tables + VACUUM + clear
   cache/session).
2. **`blueprints/query.py`** (`/query`, `/download_csv`) — takes a NL question,
   calls `generate_sql`, runs the SQL, renders results. Results over
   `MAX_TABLE_ROWS` (100) are offered as a CSV download instead of an HTML table.
   Query results (SQL + DataFrame JSON + chart suggestions) are cached by
   `table:question`.
3. **`blueprints/chart.py`** (`/generate_chart`, AJAX) — takes a NL chart
   `description` + `scope`, produces a `ChartSpec`, reshapes the current result, and
   returns an ECharts JSON payload.

**Services (`app/services/`) — the core logic:**

- `llm.py` — provider-agnostic model construction (lazy per-provider imports,
  `@cache`d) and the text-to-SQL agent. `generate_sql` runs a single structured
  PydanticAI call returning a `SqlQuery`. `run_with_backoff` retries on rate-limit /
  quota errors across providers. The SQL system prompt enforces **read-only, single
  SELECT** and chart-friendly column ordering (category/label first, measure second).
- `visualization.py` — two more agents: `analyze_visualization` suggests three short
  NL chart ideas for a result; `generate_chart_spec` turns a NL description into a
  `ChartSpec`. `apply_chart_spec` reshapes the DataFrame with pandas
  (group/aggregate/sort/limit) and `build_chart_payload` emits the per-chart-type
  JSON. Key invariant: **reshape-only** — chart specs may only reference columns
  already in the current result; `_sanitize_spec` forces any hallucinated column back
  to a real one. `_to_numeric` tolerates thousands separators and currency/percent
  symbols.
- `schemas.py` — the typed PydanticAI output contracts (`SqlQuery`, `ChartType`,
  `Aggregation`, `SortOrder`, `ChartSuggestions`, `ChartSpec`). These are the LLM's
  guardrails; changing them changes what the models are allowed to return.
- `database.py` — cached SQLAlchemy engine, name/type inference, table/schema helpers.
- `runtime.py` — a **process-global** holder for the active DataFrame (`runtime.df`).
  This is how `/query` hands the result to `/generate_chart`. It is **not
  per-session or thread-safe** — a known single-user limitation, not a bug to
  "fix" casually.

## Conventions

- **Keep comments short.** One concise line stating the *why*; no multi-line
  comment blocks explaining mechanics the code already shows.

## Things to know before changing code

- **Reshape-only charts.** The chart pipeline never fetches new data; it only
  reshapes the last query result held in `runtime.df`. Preserve this when touching
  `visualization.py` or `schemas.ChartSpec`.
- **LLM output is constrained by `schemas.py`, not free text.** To let the models
  produce something new (a new chart type, aggregation, etc.), update the schema and
  the corresponding system prompt together.
- **Single-dataset semantics are intentional.** Upload drops all tables; there is one
  active table and one active DataFrame at a time.
- **Known follow-ups** (documented in `README.md`): rotate the previously-committed
  `GOOGLE_API_KEY`, add SQL self-correction/retry, make `runtime.df` per-session, and
  run LLM-generated SQL under a read-only DB role.