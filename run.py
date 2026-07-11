"""Development entry point for the Flask app."""

import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    # Set FLASK_DEBUG=1 in .env for auto-reload on code changes during development.
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=8000, debug=debug)
