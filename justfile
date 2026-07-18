# Text to Visuals — task runner (https://just.systems)
#
# Quick start:
#   just setup     # one-time: venv + deps + .env + db migrations
#   just dev       # run the dev server on http://localhost:8000
#
# Run `just` with no argument to list all recipes.

# Use PowerShell on Windows (always present) and sh elsewhere. Every recipe
# body below is a single program call, so it runs the same under either.
set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]
set shell := ["sh", "-cu"]

venv_dir := ".venv"

# Interpreter used to CREATE the venv (Windows has the `py` launcher, not `python`).
sys_python := if os_family() == "windows" { "py" } else { "python3" }

# Interpreter INSIDE the venv, used for everything else.
python := venv_dir / (if os_family() == "windows" { "Scripts/python.exe" } else { "bin/python" })

# List available recipes.
default:
    @just --list

# One-time local setup: virtualenv, dependencies, .env file, and db migrations.
setup: _venv deps _env db
    @echo "Setup complete. Edit .env with your keys, then run 'just dev'."

# Create the virtualenv if it doesn't already exist.
_venv:
    @{{ if path_exists(venv_dir) == "true" { "echo \"Using existing " + venv_dir + "\"" } else { sys_python + " -m venv " + venv_dir } }}

# Install/upgrade Python dependencies into the venv.
deps:
    {{python}} -m pip install --upgrade pip
    {{python}} -m pip install -r requirements.txt

# Create .env from .env.example if it's missing (never overwrites yours).
_env:
    @{{ if path_exists(".env") == "true" { "echo \"Keeping existing .env\"" } else { sys_python + " -c \"import shutil; shutil.copy('.env.example', '.env'); print('Created .env from .env.example - now fill in your keys')\"" } }}

# Apply database migrations. No-op until a migration tool is wired in.
db:
    @{{ if path_exists("migrations") == "true" { python + " -m flask db upgrade" } else if path_exists("alembic.ini") == "true" { python + " -m alembic upgrade head" } else { "echo \"No database migrations configured yet - skipping.\"" } }}

# Run the development server (auto-reload if FLASK_DEBUG=1 in .env).
dev:
    {{python}} run.py

# Run the production server (gunicorn; Linux/macOS only).
serve:
    {{python}} -m gunicorn run:app --bind 0.0.0.0:8000

# Reinstall dependencies (e.g. after requirements.txt changes).
update: deps
    @echo "Dependencies updated."

# Remove the virtualenv.
clean:
    @{{ if path_exists(venv_dir) == "true" { sys_python + " -c \"import shutil; shutil.rmtree('" + venv_dir + "'); print('Removed " + venv_dir + "')\"" } else { "echo \"Nothing to clean\"" } }}