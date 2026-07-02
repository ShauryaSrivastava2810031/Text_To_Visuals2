import os

from flask import Flask

from .config import Config
from .extensions import cache


def create_app(config_class=Config):
    """Application factory."""
    app = Flask(__name__)
    app.config.from_object(config_class)

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
