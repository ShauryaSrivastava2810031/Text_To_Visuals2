import logging
import os

from flask import Flask

from .config import Config
from .extensions import cache


def _configure_logging(app):
    """Send app + LLM logs to the console at the configured level."""
    level = logging.DEBUG if app.config.get("LOG_LEVEL") == "DEBUG" else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")
    )
    t2v = logging.getLogger("t2v")
    t2v.setLevel(level)
    t2v.handlers.clear()
    t2v.addHandler(handler)
    t2v.propagate = False


def create_app(config_class=Config):
    """Application factory."""
    app = Flask(__name__)
    app.config.from_object(config_class)

    _configure_logging(app)

    # Initialize extensions
    cache.init_app(app)

    # Ensure the upload folder exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Register blueprints
    from .blueprints.main import main_bp
    from .blueprints.query import query_bp
    from .blueprints.chart import chart_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(query_bp)
    app.register_blueprint(chart_bp)

    return app
