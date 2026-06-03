"""
app/services/ml_service.py
--------------------------
Machine Learning inference service.

Responsibilities:
- Load and cache the trained model + vectorizer from disk
- Preprocess incoming text
- Return prediction label and confidence score
"""

import os
import pickle
import numpy as np
from typing import Tuple, Optional

from app.services.preprocess import clean_text


# -------------------------------------------------------------------
# Module-level cache — model is loaded once on first call
# -------------------------------------------------------------------
_model = None
_vectorizer = None


def _load_artifacts(model_path: str, vectorizer_path: str) -> bool:
    """
    Load ML model and vectorizer from disk into module-level cache.

    Args:
        model_path (str): Absolute path to model.pkl
        vectorizer_path (str): Absolute path to vectorizer.pkl

    Returns:
        bool: True if loaded successfully, False otherwise.
    """
    global _model, _vectorizer

    if _model is not None and _vectorizer is not None:
        return True  # Already loaded

    if not os.path.exists(model_path):
        return False
    if not os.path.exists(vectorizer_path):
        return False

    try:
        with open(model_path, "rb") as f:
            _model = pickle.load(f)
        with open(vectorizer_path, "rb") as f:
            _vectorizer = pickle.load(f)
        return True
    except Exception:
        _model = None
        _vectorizer = None
        return False


def predict(
    text: str,
    model_path: str,
    vectorizer_path: str,
) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """
    Run ML inference on the given text.

    Args:
        text (str): Raw user-submitted news text.
        model_path (str): Path to model.pkl
        vectorizer_path (str): Path to vectorizer.pkl

    Returns:
        Tuple of:
            label (str|None)      : "real" or "fake", or None on error
            confidence (float|None): 0.0–100.0 percentage, or None on error
            error (str|None)      : Error message if something went wrong
    """
    # Load artifacts (cached after first call)
    if not _load_artifacts(model_path, vectorizer_path):
        return None, None, "Model not loaded. Please run train_model.py first."

    # Validate input
    if not text or not text.strip():
        return None, None, "Input text is empty."

    try:
        # Step 1: Preprocess
        cleaned = clean_text(text)

        if not cleaned:
            return None, None, "Text became empty after preprocessing."

        # Step 2: TF-IDF transform
        features = _vectorizer.transform([cleaned])

        # Step 3: Predict
        prediction = _model.predict(features)[0]  # 0=fake, 1=real

        # Step 4: Confidence score
        # PassiveAggressiveClassifier uses decision_function (not predict_proba)
        decision_scores = _model.decision_function(features)[0]

        # Convert decision score to a pseudo-probability using sigmoid
        confidence_raw = float(1 / (1 + np.exp(-abs(decision_scores))))
        confidence = round(confidence_raw * 100, 2)

        # Cap confidence between 50–99% for realistic display
        confidence = max(50.0, min(99.9, confidence))

        label = "real" if prediction == 1 else "fake"
        return label, confidence, None

    except Exception as e:
        return None, None, f"Prediction error: {str(e)}"


def is_model_loaded(model_path: str, vectorizer_path: str) -> bool:
    """Check whether the model and vectorizer are available on disk."""
    return os.path.exists(model_path) and os.path.exists(vectorizer_path)
