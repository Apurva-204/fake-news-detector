"""
app/services/preprocess.py
--------------------------
Text preprocessing utilities.

IMPORTANT: clean_text() must match EXACTLY what was used during model training
(train_model.py). The training script lowercases and strips non-alpha characters
only — it does NOT remove stopwords. Using a different pipeline at prediction
time causes the TF-IDF vectors to be different from what the model learned,
leading to wrong predictions on real-world web news.
"""

import re


def clean_text(text: str) -> str:
    """
    Clean and normalize raw news text for ML inference.

    This MUST match the preprocessing used in train_model.py exactly:
        1. Lowercase all characters
        2. Remove non-alphabetic characters (keep letters + spaces only)
        3. Collapse multiple spaces

    NOTE: Stopwords are intentionally NOT removed here, because the model
    was trained without stopword removal. Removing them at inference time
    would shift the TF-IDF feature space and break predictions.

    Args:
        text (str): Raw input text (from user, OCR, or web scraping).

    Returns:
        str: Cleaned, normalized text string ready for vectorization.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove non-alpha characters (matches training script exactly)
    text = re.sub(r"[^a-zA-Z ]", "", text)

    # 3. Collapse extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
