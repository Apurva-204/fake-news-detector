"""
config.py
---------
Central configuration for the Fake News Detection System.
Loads environment variables and defines model/data paths.
"""

import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()


class Config:
    """Base configuration class."""

    # Flask settings
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    DEBUG = os.getenv("FLASK_DEBUG", "True").lower() in ("true", "1")

    # OCR Settings
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2 MB max file size
    
    # Path to Tesseract executable (Update this if installed elsewhere on your PC)
    TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # ML Model paths (relative to project root)
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.pkl")
    VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "model", "vectorizer.pkl")

    # Dataset paths
    TRUE_CSV = os.path.join(os.path.dirname(__file__), "true.csv")
    FAKE_CSV = os.path.join(os.path.dirname(__file__), "fake.csv")

    # Admin settings (simple flag — no complex auth)
    ADMIN_ENABLED = True

    # Maximum number of prediction logs to store in memory
    MAX_LOG_ENTRIES = 500

    # SQLite database path for user authentication
    DATABASE_PATH = os.path.join(os.path.dirname(__file__), "users.db")
