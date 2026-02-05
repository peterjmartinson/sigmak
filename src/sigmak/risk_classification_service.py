# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

"""
Risk classification service with similarity-first LLM caching.

This module implements the core classification logic:
1. Query cached LLM classifications via similarity search
2. If similarity >= threshold, return cached result (no LLM call)
3. If similarity < threshold, call LLM and persist result

Design Principles:
- Minimize LLM API calls through intelligent caching
- Full provenance tracking for every classification
- Configurable similarity threshold via config.yaml
- Automatic persistence to both SQLite and ChromaDB
"""

import logging
from typing import Optional, Tuple

from sigmak.config import get_settings
from sigmak.drift_detection import ClassificationSource, DriftDetectionSystem
from sigmak.embeddings import EmbeddingEngine
from sigmak.llm_classifier import GeminiClassifier, LLMClassificationResult
from sigmak.risk_taxonomy import RiskCategory
from sigmak.severity import extract_numeric_anchors, SEVERE_KEYWORDS

logger = logging.getLogger(__name__)


class RiskClassificationService:
    """
    Service for risk classification with similarity-first caching.
    
    Features:
    - Queries cached LLM classifications before calling LLM
    - Configurable similarity threshold (default 0.8)
    - Automatic persistence of all classifications
    - Full provenance tracking (prompt_version, model, timestamp)
    
    Usage:
        >>> service = RiskClassificationService()
        >>> result, source = service.classify_with_cache_first(
        ...     "Supply chain disruptions may impact operations."
        ... )
        >>> print(result.category, source)  # OPERATIONAL, 'cache' or 'llm'
    """
    
    def __init__(
        self,
        drift_system: Optional[DriftDetectionSystem] = None,
        llm_classifier: Optional[GeminiClassifier] = None,
        config_path: Optional[str] = None
    ) -> None:
        """
        Initialize the classification service.
        
        Args:
            drift_system: DriftDetectionSystem instance (creates default if None)
            llm_classifier: GeminiClassifier instance (creates default if None)
            config_path: Path to config file (uses default if None)
        """
        # Load settings
        if config_path:
            from pathlib import Path
            self.settings = get_settings(Path(config_path))
        else:
            self.settings = get_settings()
        
        # Initialize drift system (handles SQLite + ChromaDB persistence)
        if drift_system is None:
            self.drift_system = DriftDetectionSystem(
                db_path=str(self.settings.database.sqlite_path),
                chroma_path=str(self.settings.chroma.persist_directory),
                embedding_model=self.settings.chroma.embedding_model
            )
        else:
            self.drift_system = drift_system
        
        # Initialize LLM classifier
        if llm_classifier is None:
            self.llm_classifier = GeminiClassifier(
                model_name=self.settings.llm.model
            )
        else:
            self.llm_classifier = llm_classifier
        
        # Initialize embedding engine
        self.embedding_engine = EmbeddingEngine(
            model_name=self.settings.chroma.embedding_model
        )
        
        # Get similarity threshold from config
        self.similarity_threshold = self.settings.chroma.llm_cache_similarity_threshold
        
        logger.info(
            f"RiskClassificationService initialized: "
            f"threshold={self.similarity_threshold}, "
            f"model={self.settings.llm.model}"
        )
    
    def _generate_synthetic_rationale(
        self,
        text: str,
        category: RiskCategory,
        similarity_score: float,
        cached_record: dict
    ) -> str:
        """
        Generate synthetic rationale from classification metadata.
        
        Used when cached rationale is missing or for borderline similarity scores.
        
        Args:
            text: Risk paragraph text
            category: Classified risk category
            similarity_score: Vector similarity to cached classification
            cached_record: Cached classification metadata
            
        Returns:
            Synthetic rationale explaining the classification
        """
        # Extract features
        dollar_amounts = extract_numeric_anchors(text)
        severe_keywords = [kw for kw in SEVERE_KEYWORDS if kw.lower() in text.lower()]
        
        rationale_parts = [
            f"This risk is classified as {category.value} based on:",
            f"• Semantic similarity ({similarity_score:.1%}) to cached classification "
            f"from {cached_record.get('timestamp', 'unknown')[:10]}",
        ]
        
        if dollar_amounts:
            max_amount = max(dollar_amounts)
            if max_amount >= 1_000_000_000:
                amount_str = f"${max_amount / 1_000_000_000:.1f}B"
            else:
                amount_str = f"${max_amount / 1_000_000:.1f}M"
            rationale_parts.append(f"• Financial exposure: {amount_str}")
        
        if severe_keywords:
            rationale_parts.append(
                f"• Risk indicators: {', '.join(severe_keywords[:5])}"
            )
        
        # Add confidence qualifier
        if similarity_score >= 0.95:
            rationale_parts.append("• High-confidence match (near-identical risk language)")
        elif similarity_score >= 0.85:
            rationale_parts.append("• Strong semantic overlap with cached classification")
        else:
            rationale_parts.append("• Moderate similarity; based on vector database match")
        
        return "\n".join(rationale_parts)
    
    def classify_with_cache_first(
        self,
        text: str,
        force_llm: bool = False
    ) -> Tuple[LLMClassificationResult, str]:
        """
        Classify risk text using similarity-first caching.
        
        Flow:
        1. Generate embedding for input text
        2. Search cached LLM classifications in ChromaDB
        3. If top match similarity >= threshold:
           - Return cached classification (source='cache')
        4. Else:
           - Call LLM for new classification
           - Persist result to SQLite + ChromaDB
           - Return LLM classification (source='llm')
        
        Args:
            text: Risk paragraph to classify
            force_llm: If True, skip cache and always call LLM
        
        Returns:
            Tuple of (LLMClassificationResult, source_string)
            source_string is either 'cache' or 'llm'
        
        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Generate embedding for query
        embeddings = self.embedding_engine.encode([text])
        query_embedding = embeddings[0].tolist()  # Convert numpy array to list
        
        # Check cache first (unless force_llm)
        if not force_llm:
            cache_hit = self._check_cache(query_embedding, text)
            if cache_hit is not None:
                cache_result, cache_meta = cache_hit
                similarity = cache_meta.get("similarity_score")
                chroma_id = cache_meta.get("chroma_id")
                record_id = cache_meta.get("id")
                logger.info(
                    "Classification run: source=cache, "
                    f"category={cache_result.category.value}, "
                    f"confidence={cache_result.confidence:.2f}, "
                    f"similarity={similarity:.3f}, "
                    f"record_id={record_id}, chroma_id={chroma_id}"
                )
                return cache_result, "cache"
        
        # Cache miss or forced LLM: call LLM
        logger.info("Cache miss or forced LLM: calling LLM classifier")
        llm_result = self.llm_classifier.classify(text)
        
        # Persist to SQLite + ChromaDB and log whether write succeeded
        wrote_to_vector = False
        record_id = None
        chroma_id = None
        try:
            record_id, chroma_id = self.drift_system.insert_classification(
                text=text,
                embedding=query_embedding,
                llm_result=llm_result,
                source=ClassificationSource.LLM
            )
            wrote_to_vector = bool(chroma_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to persist LLM classification to storage")

        logger.info(
            "Classification run: source=llm, "
            f"category={llm_result.category.value}, "
            f"wrote_to_vector_db={'yes' if wrote_to_vector else 'no'}, "
            f"record_id={record_id}, chroma_id={chroma_id}"
        )

        return llm_result, "llm"
    
    def _check_cache(
        self,
        query_embedding: list[float],
        query_text: str
    ) -> Optional[tuple[LLMClassificationResult, dict]]:
        """
        Check cached classifications for similar text.
        
        Args:
            query_embedding: Embedding vector for query text
            query_text: Original query text (for logging)
        
        Returns:
            LLMClassificationResult if cache hit, None otherwise
        """
        # Search for similar classifications
        results = self.drift_system.similarity_search(
            query_embedding=query_embedding,
            n_results=1
        )
        
        if not results:
            logger.debug("No cached classifications found")
            return None
        
        top_result = results[0]
        similarity_score = top_result.get("similarity_score", 0.0)
        
        logger.debug(
            f"Top cached result: similarity={similarity_score:.3f}, "
            f"category={top_result.get('category')}, "
            f"threshold={self.similarity_threshold}"
        )
        
        # Check if similarity meets threshold
        if similarity_score >= self.similarity_threshold:
            # Reconstruct LLMClassificationResult from cached data
            from datetime import datetime
            
            category = RiskCategory(top_result["category"])
            cached_evidence = top_result.get("evidence", "")
            cached_rationale = top_result.get("rationale", "")
            
            # Enhance rationale based on similarity and cached data quality
            if cached_rationale and similarity_score >= 0.90:
                # Option 1: Reference-based rationale (high similarity, good cache)
                enhanced_rationale = (
                    f"Classification based on similarity to previously analyzed risk "
                    f"(similarity: {similarity_score:.1%}).\n\n"
                    f"Reference analysis: {cached_rationale}"
                )
                enhanced_evidence = cached_evidence
            elif cached_rationale and similarity_score >= 0.80:
                # Hybrid: Reference with synthetic supplement (borderline similarity)
                synthetic_part = self._generate_synthetic_rationale(
                    query_text, category, similarity_score, top_result
                )
                enhanced_rationale = (
                    f"{synthetic_part}\n\n"
                    f"Reference classification rationale: {cached_rationale[:200]}..."
                )
                enhanced_evidence = cached_evidence if cached_evidence else synthetic_part
            else:
                # Option 3: Synthetic rationale (missing cache data or low similarity)
                enhanced_rationale = self._generate_synthetic_rationale(
                    query_text, category, similarity_score, top_result
                )
                enhanced_evidence = cached_evidence if cached_evidence else enhanced_rationale
            
            reconstructed = LLMClassificationResult(
                category=category,
                confidence=top_result["confidence"],
                evidence=enhanced_evidence,
                rationale=enhanced_rationale,
                model_version=f"cache-v{top_result.get('model_version', 'unknown')}",
                prompt_version=top_result.get("prompt_version", "unknown"),
                timestamp=datetime.fromisoformat(top_result["timestamp"]),
                response_time_ms=0.0,  # Cached, no API call
                input_tokens=0,
                output_tokens=0
            )
            # Return both reconstructed result and raw metadata for logging
            return reconstructed, top_result
        
        logger.debug(f"Similarity {similarity_score:.3f} below threshold {self.similarity_threshold}")
        return None
