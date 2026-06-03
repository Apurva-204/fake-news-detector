"""
app/services/fact_check_service.py
----------------------------------
Provides live web search fact-checking using DuckDuckGo HTML search.

FIXED: Old version counted any mention of words like 'fact check', 'false',
'fake' in snippets — causing a legitimate Reuters fact-check article saying
"Claim is TRUE" to still increment debunk_score and flip real→fake.

New version uses context-aware scoring: only counts a snippet as debunking
if it contains STRONG negative framing (e.g., "X is false", "hoax", 
"fabricated", "debunked") AND the claim isn't framed as "not fake" / "verified".
The ML override only happens with score >= 3 (was 1).
"""

import re
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple

# Strong debunking signals — these strongly suggest the claim is false
STRONG_DEBUNK_PATTERNS = [
    r'\b(debunked|debunking)\b',
    r'\b(hoax|fabricated|fabrication)\b',
    r'\bfalse\s+(claim|story|report|information|news)\b',
    r'\b(claim|story|report)\s+is\s+false\b',
    r'\bno[,\s]+it\s+is\s+not\s+true\b',
    r'\bsatire\b',
    r'\bmisinformation\b',
    r'\bdisinformation\b',
    r'\bconspiracy\s+theor\b',
]

# Patterns that indicate the source is CONFIRMING the claim (should NOT penalize)
CONFIRMATION_PATTERNS = [
    r'\b(verified|confirmed|true|accurate|correct|legitimate)\b',
    r'\bnot\s+(a\s+)?(hoax|fake|false|fabricated)\b',
    r'\b(claim|story)\s+is\s+(true|accurate|verified)\b',
]

# Fact-check sites that are reliable — trust their verdict
FACT_CHECK_DOMAINS = {
    'snopes.com', 'factcheck.org', 'politifact.com', 'reuters.com/fact-check',
    'apnews.com', 'bbc.com/reality-check', 'fullfact.org', 'checkyourfact.com',
    'leadstories.com', 'boomlive.in', 'altnews.in'
}


def extract_keywords(text: str, max_words: int = 8) -> str:
    """Extract core keywords from text to form a focused search query."""
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'it',
        'that', 'this', 'has', 'have', 'had', 'be', 'been', 'being'
    }
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(keywords[:max_words])


def _score_snippet(snippet: str, title: str, url: str) -> int:
    """
    Context-aware debunk scoring for a single search result.

    Returns:
        int: 0 = no debunking signal, 1 = weak signal, 2 = strong signal
    """
    combined = (snippet + " " + title).lower()

    # Check if a reliable fact-check domain is involved
    is_fact_check_site = any(domain in url.lower() for domain in FACT_CHECK_DOMAINS)

    # Check for strong debunking language
    strong_debunk = any(re.search(p, combined) for p in STRONG_DEBUNK_PATTERNS)

    # Check for confirmation language (article is saying claim is TRUE)
    is_confirming = any(re.search(p, combined) for p in CONFIRMATION_PATTERNS)

    # If snippet explicitly confirms the claim, score = 0 regardless
    if is_confirming and not strong_debunk:
        return 0

    if strong_debunk:
        # Fact-check sites saying debunked = very strong signal
        return 2 if is_fact_check_site else 1

    return 0


def search_claim(text: str) -> Tuple[List[Dict[str, str]], int]:
    """
    Search the live web for the given claim via DuckDuckGo HTML.

    Returns:
        sources (list): Top search results with title, snippet, url, debunk_signal
        debunk_score (int): Aggregate context-aware debunk score.
                           Score >= 3 means strong evidence the claim is false.
    """
    query = extract_keywords(text)
    if not query:
        return [], 0

    try:
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            )
        }
        url = 'https://html.duckduckgo.com/html/'
        payload = {'q': query + ' fact check'}   # bias toward fact-check results

        res = requests.post(url, data=payload, headers=headers, timeout=6)
        soup = BeautifulSoup(res.text, 'html.parser')

        sources = []
        debunk_score = 0

        for a in soup.find_all('a', class_='result__snippet', limit=8):
            snippet = a.text.strip()

            result_div = a.find_parent('div', class_='result__body')
            if not result_div:
                continue

            title_heading = result_div.find('h2', class_='result__title')
            if not title_heading:
                continue

            link_el = title_heading.find('a', class_='result__a')
            if not link_el:
                continue

            title = link_el.get_text(strip=True) or "Source"
            href = link_el.get('href', '')

            # Unwrap DuckDuckGo redirect URLs
            if href.startswith('//'):
                href = 'https:' + href
            if 'duckduckgo.com/l/' in href:
                m = re.search(r'uddg=([^&]+)', href)
                if m:
                    from urllib.parse import unquote
                    href = unquote(m.group(1))

            if not href.startswith('http'):
                continue

            # Context-aware scoring (replaces naive keyword counting)
            signal = _score_snippet(snippet, title, href)
            debunk_score += signal

            sources.append({
                'title': title,
                'snippet': snippet,
                'url': href,
                'debunk_signal': signal,   # 0=neutral, 1=weak, 2=strong
            })

            if len(sources) >= 4:
                break

        return sources[:3], debunk_score

    except Exception as e:
        print(f"Web search failed: {e}")
        return [], 0
