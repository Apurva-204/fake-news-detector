"""
app/__init__.py
---------------
Flask application factory.
Creates and configures the Flask app instance.
Initializes the SQLite user database on startup.
"""

from flask import Flask
from config import Config
from app.services.auth_service import init_db


def create_app():
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )
    app.config.from_object(Config)

    # Initialize the user authentication database
    init_db(app.config["DATABASE_PATH"])

    # Register routes blueprint
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app
