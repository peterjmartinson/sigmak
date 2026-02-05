"""
Retrieval-augmented risk scoring logic for SEC filings (Issue #21).

This module computes Severity and Novelty scores for risk disclosures,
with full source citation and traceability.

Core Concepts:
- Severity: How severe is this risk? (0.0 = minor, 1.0 = catastrophic)
  - Uses sentiment-weighted scoring (VADER + quantitative anchors + keywords + novelty)
- Novelty: How new is this risk vs. historical filings? (0.0 = repetitive, 1.0 = novel)

Every score includes:
- Normalized value [0.0, 1.0]
- Source citation (exact text)
- Human-readable explanation
- Original metadata (ticker, year, etc.)
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import logging
import numpy as np
from numpy.typing import NDArray
import yaml

from sigmak.embeddings import EmbeddingEngine
from sigmak.severity import (
    compute_severity,
    get_market_cap,
    DEFAULT_WEIGHTS as SEVERITY_DEFAULT_WEIGHTS
)

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================

class ScoringError(Exception):
    """
    Raised when risk scoring fails due to invalid input or processing errors.
    
    Examples:
    - Missing required fields (text, metadata)
    - Malformed chunk structure
    - Embedding generation failure
    """
    pass


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class RiskScore:
    """
    Container for a computed risk score with full provenance.
    
    Attributes:
        value: Normalized score in [0.0, 1.0]
        source_citation: Exact text from source chunk (truncated if >500 chars)
        explanation: Human-readable description of how score was calculated
        metadata: Original chunk metadata (ticker, filing_year, item_type, etc.)
    
    Example:
        >>> score = RiskScore(
        ...     value=0.85,
        ...     source_citation="Catastrophic supply chain disruptions...",
        ...     explanation="High severity due to catastrophic language",
        ...     metadata={"ticker": "AAPL", "filing_year": 2025}
        ... )
    """
    value: float
    source_citation: str
    explanation: str
    metadata: Dict[str, Any]
    
    def __post_init__(self) -> None:
        """Validate score is in [0.0, 1.0] range."""
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Score value must be in [0.0, 1.0], got {self.value}")


# ============================================================================
# Severity Keywords
# ============================================================================

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


# ============================================================================
# Risk Scorer
# ============================================================================

class RiskScorer:
    """
    Computes Severity and Novelty scores for risk disclosures.
    
    Severity Scoring (NEW - Sentiment-Weighted):
    - Combines sentiment analysis, quantitative anchors, keyword density, and novelty
    - Uses VADER for sentiment, extracts dollar amounts, counts risk keywords
    - Normalizes by market cap when available
    - Range: [0.0, 1.0]
    
    Novelty Scoring:
    - Compares current chunk with historical embeddings
    - Measures semantic distance from past disclosures
    - Range: [0.0, 1.0], where 1.0 = maximally novel
    
    Usage:
        >>> scorer = RiskScorer()
        >>> chunk = {"text": "...", "metadata": {...}}
        >>> severity = scorer.calculate_severity(chunk, chroma_collection)
        >>> novelty = scorer.calculate_novelty(chunk, historical_chunks)
    """
    
    def __init__(
        self,
        embeddings: Optional[EmbeddingEngine] = None,
        config_path: str = "config.yaml"
    ) -> None:
        """
        Initialize risk scorer.
        
        Args:
            embeddings: Optional pre-initialized embedding engine.
                       If None, will create a new instance (lazy loading).
            config_path: Path to config YAML with severity weights
        """
        self._embeddings = embeddings
        self._severity_weights = self._load_severity_config(config_path)
    
    def _load_severity_config(self, config_path: str) -> Dict[str, float]:
        """Load severity weights from config file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get("severity", {}).get("weights", SEVERITY_DEFAULT_WEIGHTS)
        except Exception:
            logger.warning(f"Could not load config from {config_path}, using defaults")
            return SEVERITY_DEFAULT_WEIGHTS
    
    @property
    def embeddings(self) -> EmbeddingEngine:
        """Lazy-load embedding engine."""
        if self._embeddings is None:
            self._embeddings = EmbeddingEngine()
        return self._embeddings
    
    # ========================================================================
    # Severity Scoring (NEW IMPLEMENTATION)
    # ========================================================================
    
    def calculate_severity(
        self,
        chunk: Any,
        chroma_collection: Optional[Any] = None
    ) -> RiskScore:
        """
        Calculate severity score using sentiment-weighted system.
        
        NEW ALGORITHM (replaces keyword-only scoring):
        1. Extract dollar amounts and normalize by market cap
        2. Compute sentiment score (negative = high severity)
        3. Count severe/moderate keywords
        4. Measure novelty via YoY drift (requires chroma_collection)
        5. Combine with weighted formula
        
        Formula:
            severity = w_sentiment * sentiment_score +
                      w_quant * quant_anchor_score +
                      w_keyword * keyword_count_score +
                      w_novelty * novelty_score
        
        Args:
            chunk: Dictionary with 'text' and 'metadata' fields
            chroma_collection: Optional ChromaDB collection for novelty calculation
        
        Returns:
            RiskScore with severity value, citation, explanation, metadata
        
        Raises:
            ScoringError: If chunk is malformed or missing required fields
        
        Example:
            >>> chunk = {
            ...     "text": "We may face losses of $4.9B due to supply chain crisis",
            ...     "metadata": {"ticker": "BA", "filing_year": 2025}
            ... }
            >>> score = scorer.calculate_severity(chunk, chroma_collection)
            >>> assert score.value > 0.5  # High severity due to $ amount + keywords
        """
        # Validate input
        self._validate_chunk(chunk)
        
        text = chunk["text"]
        metadata = chunk["metadata"]
        
        # Handle empty text edge case
        if not text or not text.strip():
            raise ScoringError("Cannot score empty text")
        
        # Extract required metadata
        ticker = metadata.get("ticker", "UNKNOWN")
        year = metadata.get("filing_year", 2025)
        
        # Get market cap from database
        market_cap = get_market_cap(ticker)
        
        # Generate embedding for novelty calculation
        try:
            embedding = self.embeddings.encode([text])[0].tolist()
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}, using zeros")
            embedding = [0.0] * 384  # Fallback
        
        # Compute integrated severity score
        severity_value, explanation_dict = compute_severity(
            text=text,
            ticker=ticker,
            market_cap=market_cap,
            year=year,
            embedding=embedding,
            chroma_collection=chroma_collection,
            weights=self._severity_weights
        )
        
        # Build human-readable explanation
        explanation = self._build_severity_explanation(severity_value, explanation_dict)
        
        # Truncate citation for readability (max 500 chars)
        citation = text if len(text) <= 500 else text[:497] + "..."
        
        return RiskScore(
            value=severity_value,
            source_citation=citation,
            explanation=explanation,
            metadata=metadata
        )
    
    def _build_severity_explanation(
        self,
        severity: float,
        components: Dict[str, Any]
    ) -> str:
        """Build human-readable explanation from component scores."""
        dominant = components["dominant_component"]
        amounts = components["extracted_amounts"]
        
        # Format dollar amounts
        if amounts:
            max_amount = max(amounts)
            if max_amount >= 1_000_000_000:
                amount_str = f"${max_amount / 1_000_000_000:.1f}B"
            else:
                amount_str = f"${max_amount / 1_000_000:.1f}M"
        else:
            amount_str = "no quantitative anchors"
        
        # Build explanation based on severity level
        if severity >= 0.7:
            level = "High"
            desc = "indicates catastrophic or existential risk"
        elif severity >= 0.5:
            level = "Moderate-high"
            desc = "indicates significant business risk"
        elif severity >= 0.3:
            level = "Moderate"
            desc = "indicates manageable business risk"
        else:
            level = "Low"
            desc = "indicates routine business considerations"
        
        return (
            f"{level} severity (score: {severity:.2f}). "
            f"Dominant factor: {dominant} "
            f"(sentiment={components['sentiment_score']:.2f}, "
            f"quant={components['quant_anchor_score']:.2f}, "
            f"keywords={components['keyword_count_score']:.2f}, "
            f"novelty={components['novelty_score']:.2f}). "
            f"Extracted amounts: {amount_str}. "
            f"Analysis {desc}."
        )
    
    def calculate_severity_batch(
        self,
        chunks: List[Dict[str, Any]],
        chroma_collection: Optional[Any] = None
    ) -> List[RiskScore]:
        """
        Calculate severity scores for multiple chunks efficiently.
        
        Args:
            chunks: List of chunk dictionaries
            chroma_collection: Optional ChromaDB collection for novelty
        
        Returns:
            List of RiskScore objects (same order as input)
        
        Example:
            >>> chunks = [{"text": "...", "metadata": {...}}, ...]
            >>> scores = scorer.calculate_severity_batch(chunks, chroma_collection)
            >>> assert len(scores) == len(chunks)
        """
        return [self.calculate_severity(chunk, chroma_collection) for chunk in chunks]
    
    # ========================================================================
    # Novelty Scoring
    # ========================================================================
    
    def calculate_novelty(
        self,
        chunk: Dict[str, Any],
        historical_chunks: List[Dict[str, Any]]
    ) -> RiskScore:
        """
        Calculate novelty score by comparing chunk with historical filings.
        
        Algorithm:
        1. Validate inputs
        2. Handle edge case: no historical data → max novelty (1.0)
        3. Generate embeddings for current and historical chunks
        4. Compute cosine similarities between current and each historical
        5. Novelty = 1 - max(similarities)  [most similar → least novel]
        6. Generate explanation and citation
        
        Args:
            chunk: Current risk disclosure chunk
            historical_chunks: List of historical chunks for comparison
        
        Returns:
            RiskScore with novelty value, citation, explanation, metadata
        
        Raises:
            ScoringError: If chunks are malformed
        
        Example:
            >>> current = {"text": "Quantum computing threats", "metadata": {...}}
            >>> historical = [{"text": "Standard competition risks", "metadata": {...}}]
            >>> score = scorer.calculate_novelty(current, historical)
            >>> assert score.value > 0.7  # Novel topic
        """
        # Validate inputs
        self._validate_chunk(chunk)
        for hist_chunk in historical_chunks:
            self._validate_chunk(hist_chunk)
        
        text = chunk["text"]
        metadata = chunk["metadata"]
        
        # Handle empty text
        if not text or not text.strip():
            raise ScoringError("Cannot score empty text")
        
        # Edge case: no historical data → maximally novel
        if not historical_chunks:
            explanation = (
                "Maximum novelty (score: 1.00). "
                "No historical data available for comparison. "
                "This risk disclosure has no precedent in prior filings."
            )
            citation = text if len(text) <= 500 else text[:497] + "..."
            return RiskScore(
                value=1.0,
                source_citation=citation,
                explanation=explanation,
                metadata=metadata
            )
        
        # Generate embeddings
        try:
            current_embedding = self.embeddings.encode([text])[0]
            historical_texts = [h["text"] for h in historical_chunks]
            historical_embeddings = self.embeddings.encode(historical_texts)
        except Exception as e:
            raise ScoringError(f"Embedding generation failed: {e}")
        
        # Compute cosine similarities with all historical chunks
        similarities = self._compute_cosine_similarities(
            current_embedding,
            historical_embeddings
        )
        
        # Novelty = 1 - max_similarity (most similar = least novel)
        max_similarity = float(np.max(similarities))
        novelty_value = 1.0 - max_similarity
        
        # Clamp to [0.0, 1.0]
        novelty_value = max(0.0, min(1.0, novelty_value))
        
        # Generate explanation
        explanation = self._generate_novelty_explanation(
            novelty_value,
            max_similarity,
            len(historical_chunks)
        )
        
        # Truncate citation
        citation = text if len(text) <= 500 else text[:497] + "..."
        
        return RiskScore(
            value=novelty_value,
            source_citation=citation,
            explanation=explanation,
            metadata=metadata
        )
    
    def _compute_cosine_similarities(
        self,
        current_embedding: NDArray[np.float32],
        historical_embeddings: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """
        Compute cosine similarities between current and historical embeddings.
        
        Args:
            current_embedding: 1D array (384 dimensions)
            historical_embeddings: 2D array (N x 384 dimensions)
        
        Returns:
            1D array of cosine similarities (N values)
        """
        # Normalize embeddings
        current_norm = current_embedding / np.linalg.norm(current_embedding)
        historical_norms = historical_embeddings / np.linalg.norm(
            historical_embeddings, axis=1, keepdims=True
        )
        
        # Compute dot products (cosine similarity for normalized vectors)
        similarities: NDArray[np.float32] = np.dot(historical_norms, current_norm)
        
        return similarities
    
    def _generate_novelty_explanation(
        self,
        novelty: float,
        max_similarity: float,
        historical_count: int
    ) -> str:
        """Generate human-readable explanation for novelty score."""
        if novelty >= 0.8:
            return (
                f"High novelty (score: {novelty:.2f}). "
                f"Semantically distant from {historical_count} historical chunks "
                f"(max similarity: {max_similarity:.2f}). "
                f"This risk represents a significant departure from prior disclosures."
            )
        elif novelty >= 0.5:
            return (
                f"Moderate novelty (score: {novelty:.2f}). "
                f"Partially distinct from {historical_count} historical chunks "
                f"(max similarity: {max_similarity:.2f}). "
                f"This risk has some novel aspects compared to prior filings."
            )
        elif novelty >= 0.2:
            return (
                f"Low novelty (score: {novelty:.2f}). "
                f"Similar to {historical_count} historical chunks "
                f"(max similarity: {max_similarity:.2f}). "
                f"This risk closely resembles prior year disclosures."
            )
        else:
            return (
                f"Minimal novelty (score: {novelty:.2f}). "
                f"Nearly identical to {historical_count} historical chunks "
                f"(max similarity: {max_similarity:.2f}). "
                f"This risk is repetitive boilerplate language."
            )
    
    # ========================================================================
    # Validation Helpers
    # ========================================================================
    
    def _validate_chunk(self, chunk: Any) -> None:
        """
        Validate chunk structure.
        
        Raises:
            ScoringError: If chunk is malformed
        """
        if not isinstance(chunk, dict):
            raise ScoringError(
                f"Chunk must be a dictionary, got {type(chunk).__name__}"
            )
        
        if "text" not in chunk:
            raise ScoringError(
                "Chunk missing required 'text' field. "
                f"Available keys: {list(chunk.keys())}"
            )
        
        if "metadata" not in chunk:
            raise ScoringError(
                "Chunk missing required 'metadata' field. "
                f"Available keys: {list(chunk.keys())}"
            )
        
        if not isinstance(chunk["metadata"], dict):
            raise ScoringError(
                f"Metadata must be a dictionary, got {type(chunk['metadata']).__name__}"
            )
