"""
Unit tests for the new sentiment-weighted severity scoring system.

Tests follow SRP: Each test verifies exactly one behavior.
"""
import pytest
from typing import Dict, List, Optional
from unittest.mock import Mock, patch, MagicMock


@pytest.fixture
def mock_chroma_collection():
    """Mock ChromaDB collection for novelty testing."""
    mock_collection = Mock()
    mock_collection.query.return_value = {
        "distances": [[0.3]],  # 0.3 similarity = 0.7 novelty
        "documents": [["previous risk text"]],
        "metadatas": [[{"year": 2024}]]
    }
    return mock_collection


class TestExtractNumericAnchors:
    """Test extraction of dollar amounts from risk text."""
    
    def test_extract_single_dollar_amount(self):
        """Should extract one dollar amount correctly."""
        from sigmak.severity import extract_numeric_anchors
        
        text = "We may incur losses up to $4.9 billion in 2025."
        result = extract_numeric_anchors(text)
        
        assert len(result) == 1
        assert result[0] == 4_900_000_000.0
    
    def test_extract_multiple_dollar_amounts(self):
        """Should extract all dollar amounts and return maximum."""
        from sigmak.severity import extract_numeric_anchors
        
        text = "Losses of $4.9B in 2025 and $3.5B in 2024."
        result = extract_numeric_anchors(text)
        
        assert len(result) == 2
        assert max(result) == 4_900_000_000.0
    
    def test_extract_million_notation(self):
        """Should handle 'million' notation."""
        from sigmak.severity import extract_numeric_anchors
        
        text = "Potential liability of $350 million."
        result = extract_numeric_anchors(text)
        
        assert result[0] == 350_000_000.0
    
    def test_extract_no_amounts(self):
        """Should return empty list when no amounts present."""
        from sigmak.severity import extract_numeric_anchors
        
        text = "This is a risk with no monetary value."
        result = extract_numeric_anchors(text)
        
        assert result == []
    
    def test_extract_handles_commas(self):
        """Should parse dollar amounts with comma separators."""
        from sigmak.severity import extract_numeric_anchors
        
        text = "Total exposure: $1,234,567."
        result = extract_numeric_anchors(text)
        
        assert result[0] == 1_234_567.0


class TestComputeSentimentScore:
    """Test sentiment scoring with negative sentiment increasing severity."""
    
    def test_negative_sentiment_increases_score(self):
        """Should return high score for negative sentiment text."""
        from sigmak.severity import compute_sentiment_score
        
        text = "catastrophic failure devastating collapse crisis"
        score = compute_sentiment_score(text)
        
        assert score > 0.5  # Negative sentiment = high severity
        assert 0.0 <= score <= 1.0
    
    def test_positive_sentiment_decreases_score(self):
        """Should return low score for positive sentiment text."""
        from sigmak.severity import compute_sentiment_score
        
        text = "excellent opportunities strong growth beneficial improvements"
        score = compute_sentiment_score(text)
        
        assert score < 0.5  # Positive sentiment = low severity
        assert 0.0 <= score <= 1.0
    
    def test_neutral_sentiment_middle_score(self):
        """Should return mid-range score for neutral text."""
        from sigmak.severity import compute_sentiment_score
        
        text = "The company operates in multiple markets."
        score = compute_sentiment_score(text)
        
        assert 0.3 <= score <= 0.7
    
    def test_empty_text_returns_neutral(self):
        """Should handle empty text gracefully."""
        from sigmak.severity import compute_sentiment_score
        
        score = compute_sentiment_score("")
        
        assert score == 0.5  # Neutral default


class TestComputeQuantAnchorScore:
    """Test quantitative anchor scoring with market cap normalization."""
    
    def test_normalize_by_market_cap(self):
        """Should normalize dollar amounts by market cap."""
        from sigmak.severity import compute_quant_anchor_score
        
        amounts = [1_000_000_000.0]  # $1B
        market_cap = 10_000_000_000.0  # $10B
        
        score = compute_quant_anchor_score(amounts, market_cap)
        
        # $1B / $10B = 0.1 (10% of market cap)
        assert 0.0 <= score <= 1.0
        assert score > 0.0  # Should be non-zero
    
    def test_large_amount_vs_small_cap(self):
        """Should return high score when amount exceeds market cap."""
        from sigmak.severity import compute_quant_anchor_score
        
        amounts = [5_000_000_000.0]  # $5B
        market_cap = 1_000_000_000.0  # $1B
        
        score = compute_quant_anchor_score(amounts, market_cap)
        
        assert score > 0.5  # 500% of cap = high severity
    
    def test_no_market_cap_uses_log_fallback(self):
        """Should use log normalization when market cap unavailable."""
        from sigmak.severity import compute_quant_anchor_score
        
        amounts = [1_000_000_000.0]  # $1B
        market_cap = None
        
        score = compute_quant_anchor_score(amounts, market_cap)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.0
    
    def test_empty_amounts_returns_zero(self):
        """Should return 0.0 when no amounts extracted."""
        from sigmak.severity import compute_quant_anchor_score
        
        score = compute_quant_anchor_score([], 10_000_000_000.0)
        
        assert score == 0.0
    
    def test_uses_maximum_amount(self):
        """Should prioritize maximum value from multiple amounts."""
        from sigmak.severity import compute_quant_anchor_score
        
        amounts = [1_000_000_000.0, 5_000_000_000.0, 2_000_000_000.0]
        market_cap = 10_000_000_000.0
        
        score = compute_quant_anchor_score(amounts, market_cap)
        
        # Should be based on $5B (max), not average
        assert score > 0.3


class TestComputeKeywordCountScore:
    """Test keyword density scoring."""
    
    def test_high_keyword_density(self):
        """Should return high score for many severe keywords."""
        from sigmak.severity import compute_keyword_count_score
        
        text = "catastrophic failure critical crisis severe threat devastating collapse"
        score = compute_keyword_count_score(text)
        
        assert score > 0.5
        assert 0.0 <= score <= 1.0
    
    def test_low_keyword_density(self):
        """Should return low score for few keywords."""
        from sigmak.severity import compute_keyword_count_score
        
        text = "The company operates in various markets with standard procedures."
        score = compute_keyword_count_score(text)
        
        assert score < 0.3
    
    def test_normalizes_by_text_length(self):
        """Should normalize keyword count by word count."""
        from sigmak.severity import compute_keyword_count_score
        
        short_text = "critical severe crisis"
        long_text = "critical severe crisis " + " ".join(["normal"] * 1000)
        
        score_short = compute_keyword_count_score(short_text)
        score_long = compute_keyword_count_score(long_text)
        
        # Short text has higher keyword density (3 keywords in 3 words vs 3 in 1003 words)
        assert score_short > score_long
    
    def test_empty_text_returns_zero(self):
        """Should return 0.0 for empty text."""
        from sigmak.severity import compute_keyword_count_score
        
        score = compute_keyword_count_score("")
        
        assert score == 0.0


class TestComputeNoveltyScore:
    """Test novelty scoring against previous year's embeddings."""
    
    def test_high_novelty_new_risk(self, mock_chroma_collection):
        """Should return high novelty for dissimilar risk."""
        from sigmak.severity import compute_novelty_score
        
        # ChromaDB cosine distance: 0 = identical, 2 = orthogonal
        # High distance (0.9) = dissimilar = high novelty
        mock_chroma_collection.query.return_value = {
            "distances": [[0.9]],  # Distance 0.9 out of max 2.0
        }
        
        current_embedding = [0.1] * 384
        score = compute_novelty_score(current_embedding, "AAPL", 2025, mock_chroma_collection)
        
        # novelty = distance / 2.0 = 0.9 / 2.0 = 0.45
        assert score == 0.45
        assert 0.0 <= score <= 1.0
    
    def test_low_novelty_similar_risk(self, mock_chroma_collection):
        """Should return low novelty for similar risk."""
        from sigmak.severity import compute_novelty_score
        
        mock_chroma_collection.query.return_value = {
            "distances": [[0.1]],  # Very similar
        }
        
        current_embedding = [0.1] * 384
        score = compute_novelty_score(current_embedding, "AAPL", 2025, mock_chroma_collection)
        
        assert score < 0.3
    
    def test_no_previous_year_returns_max_novelty(self, mock_chroma_collection):
        """Should return 1.0 novelty when no previous filing exists."""
        from sigmak.severity import compute_novelty_score
        
        mock_chroma_collection.query.return_value = {
            "distances": [[]],  # No results
        }
        
        current_embedding = [0.1] * 384
        score = compute_novelty_score(current_embedding, "NEWCO", 2025, mock_chroma_collection)
        
        assert score == 1.0  # Completely novel


class TestComputeSeverity:
    """Test integrated severity calculation with all components."""
    
    def test_severity_calculation_all_components(self, mock_chroma_collection):
        """Should compute weighted severity with all components."""
        from sigmak.severity import compute_severity
        
        text = "We face catastrophic losses up to $4.9 billion in critical operations."
        ticker = "BA"
        market_cap = 100_000_000_000.0  # $100B
        year = 2025
        embedding = [0.1] * 384
        
        severity, explanation = compute_severity(
            text=text,
            ticker=ticker,
            market_cap=market_cap,
            year=year,
            embedding=embedding,
            chroma_collection=mock_chroma_collection
        )
        
        assert 0.0 <= severity <= 1.0
        assert "sentiment_score" in explanation
        assert "quant_anchor_score" in explanation
        assert "keyword_count_score" in explanation
        assert "novelty_score" in explanation
        assert "extracted_amounts" in explanation
    
    def test_severity_weights_sum_correctly(self, mock_chroma_collection):
        """Should apply default weights correctly."""
        from sigmak.severity import compute_severity
        
        text = "severe risk crisis critical"
        severity, explanation = compute_severity(
            text=text,
            ticker="TEST",
            market_cap=1_000_000_000.0,
            year=2025,
            embedding=[0.1] * 384,
            chroma_collection=mock_chroma_collection
        )
        
        # Verify weights sum to severity (approximately)
        weighted_sum = (
            explanation["sentiment_score"] * 0.45 +
            explanation["quant_anchor_score"] * 0.35 +
            explanation["keyword_count_score"] * 0.10 +
            explanation["novelty_score"] * 0.10
        )
        
        assert abs(severity - weighted_sum) < 0.01
    
    def test_severity_with_no_market_cap(self, mock_chroma_collection):
        """Should handle missing market cap gracefully."""
        from sigmak.severity import compute_severity
        
        text = "We may face losses of $500 million."
        severity, explanation = compute_severity(
            text=text,
            ticker="PRIVATE",
            market_cap=None,
            year=2025,
            embedding=[0.1] * 384,
            chroma_collection=mock_chroma_collection
        )
        
        assert 0.0 <= severity <= 1.0
        assert explanation["quant_anchor_score"] >= 0.0  # Fallback used
    
    def test_explanation_includes_extracted_amounts(self, mock_chroma_collection):
        """Should include extracted dollar amounts in explanation."""
        from sigmak.severity import compute_severity
        
        text = "Potential losses: $4.9B in 2025 and $3.5B in 2024."
        _, explanation = compute_severity(
            text=text,
            ticker="BA",
            market_cap=100_000_000_000.0,
            year=2025,
            embedding=[0.1] * 384,
            chroma_collection=mock_chroma_collection
        )
        
        assert len(explanation["extracted_amounts"]) == 2
        assert 4_900_000_000.0 in explanation["extracted_amounts"]
        assert 3_500_000_000.0 in explanation["extracted_amounts"]
