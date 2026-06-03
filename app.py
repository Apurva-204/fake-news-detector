"""
app.py
------
Application entry point.

Run with:
    python app.py
    OR: flask run
"""

from app import create_app

app = create_app()

if __name__ == "__main__":
    print("=" * 55)
    print("  Fake News Detection System")
    print("  http://127.0.0.1:5000")
    print("=" * 55)
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=app.config.get("DEBUG", True),
    )
