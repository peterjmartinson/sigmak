"""
Sentiment-weighted severity scoring for SEC filing risk factors.

This module implements a configurable, explainable severity calculation that uses:
- Sentiment analysis (VADER)
- Quantitative anchors (dollar amounts normalized by market cap)
- Keyword density (severe/moderate risk terms)
- Novelty scoring (YoY drift via vector similarity)

Formula:
    severity = w_sentiment * sentiment_score +
               w_quant * quant_anchor_score +
               w_count * keyword_count_score +
               w_novelty * novelty_score

Default weights: sentiment=0.45, quant_anchor=0.35, keyword_count=0.10, novelty=0.10
"""
import re
import math
import sqlite3
from typing import List, Dict, Tuple, Optional, Any
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Keywords indicating high severity risks (catastrophic/existential language)
SEVERE_KEYWORDS = [
    "catastrophic", "existential", "unprecedented", "severe", "critical",
    "devastating", "collapse", "failure", "crisis", "threat", "significant",
    "material", "substantial", "major", "adversely", "harm", "damage",
    "disrupt", "impair", "unable", "bankruptcy", "insolvency", "default"
]

# Keywords indicating moderate severity
MODERATE_KEYWORDS = [
    "challenge", "difficulty", "risk", "uncertain", "volatility",
    "fluctuation", "pressure", "competition", "impact", "affect",
    "change", "regulatory", "compliance", "litigation"
]


# Default scoring weights (can be overridden via config)
DEFAULT_WEIGHTS = {
    "sentiment": 0.45,
    "quant_anchor": 0.35,
    "keyword_count": 0.10,
    "novelty": 0.10,
}

# Initialize VADER sentiment analyzer
_sentiment_analyzer = SentimentIntensityAnalyzer()


def extract_numeric_anchors(text: str) -> List[float]:
    """
    Extract dollar amounts from risk text.
    
    Supports formats:
    - $4.9 billion, $4.9B
    - $350 million, $350M
    - $1,234,567
    
    Returns maximum value when multiple amounts present (per Boeing requirement).
    
    Args:
        text: Risk factor text
        
    Returns:
        List of extracted dollar amounts (in dollars)
    """
    amounts: List[float] = []
    
    # Pattern: $XXX billion/B or $XXX million/M
    pattern_b_m = r'\$\s*(\d+(?:\.\d+)?)\s*(billion|B\b|million|M\b)'
    matches = re.finditer(pattern_b_m, text, re.IGNORECASE)
    
    for match in matches:
        value = float(match.group(1))
        unit = match.group(2).lower()
        
        if unit in ('billion', 'b'):
            amounts.append(value * 1_000_000_000)
        elif unit in ('million', 'm'):
            amounts.append(value * 1_000_000)
    
    # Pattern: $X,XXX,XXX (with optional commas)
    pattern_numeric = r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)'
    matches = re.finditer(pattern_numeric, text)
    
    for match in matches:
        value_str = match.group(1).replace(',', '')
        value = float(value_str)
        # Only include if >= $1M to avoid noise from small amounts
        if value >= 1_000_000:
            amounts.append(value)
    
    return amounts


def compute_sentiment_score(text: str) -> float:
    """
    Compute sentiment score where negative sentiment increases severity.
    
    Uses VADER sentiment analyzer. Converts compound score [-1, 1] to [0, 1]
    where 1.0 = maximum negative sentiment (high risk severity).
    
    Args:
        text: Risk factor text
        
    Returns:
        Sentiment score in range [0.0, 1.0], where higher = more negative
    """
    if not text.strip():
        return 0.5  # Neutral default
    
    scores = _sentiment_analyzer.polarity_scores(text)
    compound = scores['compound']  # Range: [-1, 1]
    
    # Convert: -1 (negative) -> 1.0 (high severity)
    #          +1 (positive) -> 0.0 (low severity)
    #           0 (neutral)  -> 0.5 (medium severity)
    sentiment_score = (1.0 - compound) / 2.0
    
    return sentiment_score


def compute_quant_anchor_score(
    amounts: List[float],
    market_cap: Optional[float]
) -> float:
    """
    Compute quantitative anchor score normalized by market cap.
    
    When market cap available: ratio of max amount to market cap, capped at 1.0
    When market cap unavailable: log normalization
    
    Args:
        amounts: List of extracted dollar amounts
        market_cap: Company market cap in dollars (or None)
        
    Returns:
        Quantitative score in range [0.0, 1.0]
    """
    if not amounts:
        return 0.0
    
    max_amount = max(amounts)
    
    if market_cap and market_cap > 0:
        # Normalize by market cap: $5B loss / $10B cap = 0.5 severity
        ratio = max_amount / market_cap
        # Cap at 1.0 (amounts exceeding cap = maximum severity)
        return min(ratio, 1.0)
    else:
        # Fallback: log normalization
        # log10($1M) = 6, log10($1B) = 9, log10($1T) = 12
        # Normalize to [0, 1] range: (log - 6) / 6 where 6-12 = typical range
        log_amount = math.log10(max(max_amount, 1.0))
        normalized = (log_amount - 6.0) / 6.0
        return max(0.0, min(normalized, 1.0))


def compute_keyword_count_score(text: str) -> float:
    """
    Compute keyword density score based on severe/moderate risk terms.
    
    Counts keyword matches and normalizes by text length (per 1K words).
    
    Args:
        text: Risk factor text
        
    Returns:
        Keyword score in range [0.0, 1.0]
    """
    if not text.strip():
        return 0.0
    
    text_lower = text.lower()
    word_count = len(text.split())
    
    # Count severe and moderate keywords
    severe_count = sum(1 for kw in SEVERE_KEYWORDS if kw in text_lower)
    moderate_count = sum(1 for kw in MODERATE_KEYWORDS if kw in text_lower)
    
    # Weight severe keywords more heavily
    weighted_count = severe_count * 2.0 + moderate_count * 1.0
    
    # Normalize per 1000 words
    if word_count == 0:
        return 0.0
    
    density = (weighted_count / word_count) * 1000.0
    
    # Divide by 20 (default divisor from spec) and cap at 1.0
    score = density / 20.0
    return min(score, 1.0)


def compute_novelty_score(
    embedding: List[float],
    ticker: str,
    year: int,
    chroma_collection: Any
) -> float:
    """
    Compute novelty score via YoY drift detection.
    
    Compares current risk embedding against previous year's filings for same company.
    Uses ChromaDB vector similarity.
    
    Args:
        embedding: Current risk paragraph embedding
        ticker: Company ticker symbol
        year: Current filing year
        chroma_collection: ChromaDB collection instance
        
    Returns:
        Novelty score in range [0.0, 1.0], where 1.0 = completely novel
    """
    try:
        # Query for previous year's risks from same company
        results = chroma_collection.query(
            query_embeddings=[embedding],
            n_results=1,
            where={
                "$and": [
                    {"ticker": ticker},
                    {"year": year - 1}
                ]
            }
        )
        
        distances = results.get("distances", [[]])
        
        if not distances or not distances[0]:
            # No previous year data = completely novel
            return 1.0
        
        # ChromaDB distance: 0 = identical, 2 = orthogonal
        # Convert to novelty: low distance = low novelty
        distance = distances[0][0]
        novelty = min(distance / 2.0, 1.0)  # Normalize to [0, 1]
        
        return novelty
        
    except Exception:
        # If query fails, assume maximum novelty (conservative)
        return 1.0


def get_market_cap(ticker: str, db_path: str = "database/sec_filings.db") -> Optional[float]:
    """
    Retrieve market cap for ticker from SQLite database.
    
    Args:
        ticker: Company ticker symbol
        db_path: Path to SQLite database
        
    Returns:
        Market cap in dollars, or None if not found
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT market_cap FROM peers WHERE ticker = ? LIMIT 1",
            (ticker.upper(),)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0]:
            return float(result[0])
        
        return None
        
    except Exception:
        return None


def compute_severity(
    text: str,
    ticker: str,
    market_cap: Optional[float],
    year: int,
    embedding: List[float],
    chroma_collection: Any,
    weights: Optional[Dict[str, float]] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute integrated severity score with explainable components.
    
    Combines sentiment, quantitative anchors, keyword density, and novelty
    into weighted severity score.
    
    Args:
        text: Risk factor text
        ticker: Company ticker symbol
        market_cap: Market capitalization (or None for fallback)
        year: Filing year
        embedding: Risk paragraph embedding vector
        chroma_collection: ChromaDB collection for novelty calculation
        weights: Optional custom weights (defaults to DEFAULT_WEIGHTS)
        
    Returns:
        Tuple of (severity_score, explanation_dict)
        - severity_score: float in [0.0, 1.0]
        - explanation_dict: component scores and extracted values
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    
    # Extract components
    amounts = extract_numeric_anchors(text)
    sentiment_score = compute_sentiment_score(text)
    quant_anchor_score = compute_quant_anchor_score(amounts, market_cap)
    keyword_count_score = compute_keyword_count_score(text)
    novelty_score = compute_novelty_score(embedding, ticker, year, chroma_collection)
    
    # Compute weighted severity
    severity = (
        weights["sentiment"] * sentiment_score +
        weights["quant_anchor"] * quant_anchor_score +
        weights["keyword_count"] * keyword_count_score +
        weights["novelty"] * novelty_score
    )
    
    # Build explanation dictionary
    explanation = {
        "sentiment_score": sentiment_score,
        "quant_anchor_score": quant_anchor_score,
        "keyword_count_score": keyword_count_score,
        "novelty_score": novelty_score,
        "extracted_amounts": amounts,
        "weights_used": weights,
        "dominant_component": max(
            [
                ("sentiment", sentiment_score * weights["sentiment"]),
                ("quant_anchor", quant_anchor_score * weights["quant_anchor"]),
                ("keyword_count", keyword_count_score * weights["keyword_count"]),
                ("novelty", novelty_score * weights["novelty"]),
            ],
            key=lambda x: x[1]
        )[0]
    }
    
    return severity, explanation
