"""
app/services/ocr_service.py
---------------------------
Service for extracting text from images using EasyOCR.
It includes preprocessing (resizing and grayscaling) to improve accuracy.
"""

import os
import re
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Initialize EasyOCR reader globally
try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False)
except ImportError:
    logger.warning("easyocr is not installed. OCR will fail.")
    reader = None

def preprocess_image(filepath: str) -> np.ndarray:
    """
    Open an image, resize it if it's too large (maintaining aspect ratio),
    convert it to grayscale, and return it as a numpy array.
    """
    img = Image.open(filepath)

    # Convert to grayscale
    img = img.convert('L')

    # Resize using thumbnail to maintain aspect ratio with max dimension 800px
    img.thumbnail((800, 800))
    
    # Convert PIL Image to numpy array (required by EasyOCR)
    return np.array(img)

def clean_extracted_text(text: str) -> str:
    """
    Clean the extracted text by removing excessive whitespace and limiting to 3000 chars.
    """
    # Replace multiple spaces or newlines with a single space
    cleaned = re.sub(r'\s+', ' ', text).strip()
    
    # Limit to 3000 characters
    if len(cleaned) > 3000:
        cleaned = cleaned[:3000]
    
    return cleaned

def extract_text_from_image(filepath: str) -> str:
    """
    Extract text from an image file using EasyOCR after preprocessing.
    Returns the extracted text or an empty string on failure.
    """
    if reader is None:
        return ""

    try:
        # Preprocess the image
        img_array = preprocess_image(filepath)

        # Run OCR
        # detail=0 returns a list of strings instead of bounding boxes and confidence scores
        results = reader.readtext(img_array, detail=0)

        # Merge results properly
        extracted = " ".join(results)

        # Clean and return
        return clean_extracted_text(extracted)

    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return ""
