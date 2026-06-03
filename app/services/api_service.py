"""
app/services/api_service.py
---------------------------
Perplexity API integration for secondary news verification.

Responsibilities:
- Call Perplexity API with the news text
- Parse the response into a structured result
- Handle timeouts, failures, and invalid responses gracefully

Note:
    This is a secondary verification layer. The ML model is the
    primary decision maker. If this service fails, the system
    gracefully falls back to ML-only results.
"""

import os
import requests
from typing import Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# API endpoint and model
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
PERPLEXITY_MODEL = "sonar"


def _build_prompt(news_text: str) -> str:
    """
    Build the verification prompt sent to Perplexity API.

    Args:
        news_text (str): The raw news text to verify.

    Returns:
        str: Formatted prompt string.
    """
    return (
        f"Verify this news and provide a short explanation. "
        f"Clearly state at the beginning whether it is VERIFIED (true), "
        f"UNVERIFIED (false/misleading), or UNCERTAIN. "
        f"Limit your response to 3–4 sentences.\n\n"
        f"News Text: {news_text[:2000]}"  # Truncate to avoid token limits
    )


def verify_with_api(
    news_text: str,
    api_key: Optional[str] = None,
    timeout: int = 15,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Verify news text using the Perplexity API.

    Args:
        news_text (str): Raw news text from the user.
        api_key (str|None): Perplexity API key. Loaded from env if None.
        timeout (int): Request timeout in seconds.

    Returns:
        Tuple of:
            status (str|None)      : "verified", "not_verified", or "uncertain"
            explanation (str|None) : Short explanation from the API
            error (str|None)       : Error message if the call failed
    """
    # Resolve API key
    key = api_key or os.getenv("PERPLEXITY_API_KEY", "")

    if not key or key == "your_perplexity_api_key_here":
        return None, None, "API key not configured."

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": PERPLEXITY_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a professional fact-checker. Your job is to verify "
                    "whether a given news item is accurate. Always begin your "
                    "response with one of: VERIFIED, UNVERIFIED, or UNCERTAIN."
                ),
            },
            {
                "role": "user",
                "content": _build_prompt(news_text),
            },
        ],
        "max_tokens": 300,
        "temperature": 0.2,
    }

    try:
        response = requests.post(
            PERPLEXITY_API_URL,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()

    except requests.exceptions.Timeout:
        return None, None, "API request timed out. Showing ML result only."
    except requests.exceptions.ConnectionError:
        return None, None, "Could not connect to Perplexity API."
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else "unknown"
        return None, None, f"API returned HTTP {status_code} error."
    except Exception as e:
        return None, None, f"Unexpected API error: {str(e)}"

    # Parse response
    try:
        content = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        return None, None, "Unexpected API response format."

    if not content:
        return None, None, "Empty response from API."

    # Determine verification status from content keywords
    content_upper = content.upper()
    if content_upper.startswith("VERIFIED") or "VERIFIED" in content_upper[:30]:
        status = "verified"
    elif (
        content_upper.startswith("UNVERIFIED")
        or "UNVERIFIED" in content_upper[:30]
        or "FALSE" in content_upper[:30]
        or "FAKE" in content_upper[:30]
    ):
        status = "not_verified"
    else:
        status = "uncertain"

    return status, content, None
