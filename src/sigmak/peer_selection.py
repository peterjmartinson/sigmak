"""Peer selection and semantic comparison utilities.

Implements:
- `validate_peer_group` to filter candidate peers by market cap (yfinance-backed when available).
- simple semantic similarity helpers with a configurable backend ("semantic" uses
  sentence-transformers when available; "tfidf" uses scikit-learn TF-IDF).

The module is defensive: heavy-model dependencies are imported lazily and a
`backend` parameter (or the CLI-level flag) may be used to avoid them.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import re

logger = logging.getLogger(__name__)


def _try_import_yfinance():
    try:
        import yfinance as yf  # type: ignore

        return yf
    except Exception:
        return None


def _try_import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        return SentenceTransformer
    except Exception:
        return None


def _try_import_sklearn_tfidf():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

        return TfidfVectorizer
    except Exception:
        return None


def validate_peer_group(target_ticker: str, peer_tickers: List[str]) -> List[str]:
    """Return a validated, ordered list of peer tickers for `target_ticker`.

    Rules:
    - Query marketCap via `yfinance` when available. Keep peers with marketCap
      in [0.1 * target, 10 * target].
    - Fallback: if filtered set is empty, prefer peers that share SIC with
      target and return top 10 by marketCap. If SIC unavailable, return top
      10 by marketCap among `peer_tickers`.

    This function is conservative and logs warnings if marketCap is missing.
    """
    yf = _try_import_yfinance()
    market_caps: Dict[str, Optional[float]] = {}
    sics: Dict[str, Optional[str]] = {}

    def _fetch_info(ticker: str) -> None:
        if ticker in market_caps:
            return
        if not yf:
            market_caps[ticker] = None
            sics[ticker] = None
            return
        try:
            tk = yf.Ticker(ticker)
            info = tk.info or {}
            market_caps[ticker] = info.get("marketCap")
            sics[ticker] = info.get("industry") or info.get("sector")
        except Exception:
            logger.warning("yfinance failed for %s", ticker)
            market_caps[ticker] = None
            sics[ticker] = None

    # Fetch target + peers
    _fetch_info(target_ticker)
    for t in peer_tickers:
        _fetch_info(t)

    target_cap = market_caps.get(target_ticker)
    # If no marketCap available for target, fall back to returning input peers
    if not target_cap:
        logger.warning("Target marketCap missing for %s; returning peers as-is", target_ticker)
        return peer_tickers[:]

    low, high = 0.1 * target_cap, 10 * target_cap
    filtered = [t for t in peer_tickers if market_caps.get(t) and low <= market_caps[t] <= high]

    if filtered:
        # Order by marketCap descending
        filtered.sort(key=lambda x: market_caps.get(x) or 0, reverse=True)
        logger.debug("Peer validation: %s -> %s", target_ticker, filtered)
        return filtered

    # Fallback: try to select peers by SIC
    target_sic = sics.get(target_ticker)
    if target_sic:
        same_sic = [t for t in peer_tickers if sics.get(t) == target_sic and market_caps.get(t)]
        if same_sic:
            same_sic.sort(key=lambda x: market_caps.get(x) or 0, reverse=True)
            return same_sic[:10]

    # Final fallback: top 10 by market cap among provided peers (ignoring Nones)
    ranked = [t for t in peer_tickers if market_caps.get(t)]
    ranked.sort(key=lambda x: market_caps.get(x) or 0, reverse=True)
    return ranked[:10]


def embed_paragraphs(paragraphs: List[str], backend: str = "semantic") -> Any:
    """Return embeddings for paragraphs using the requested backend.

    - backend='semantic': uses sentence-transformers if available.
    - backend='tfidf': uses scikit-learn TfidfVectorizer and returns dense vectors.
    """
    if backend == "tfidf":
        TfidfVectorizer = _try_import_sklearn_tfidf()
        if not TfidfVectorizer:
            raise RuntimeError("TF-IDF backend requested but scikit-learn not available")
        vec = TfidfVectorizer(ngram_range=(1, 3), min_df=1)
        mat = vec.fit_transform(paragraphs)
        return mat.toarray()

    # semantic backend
    ST = _try_import_sentence_transformers()
    if not ST:
        logger.warning("sentence-transformers not available; falling back to TF-IDF")
        return embed_paragraphs(paragraphs, backend="tfidf")
    model = ST("all-mpnet-base-v2")
    return model.encode(paragraphs, show_progress_bar=False)


def compute_semantic_similarity(a: str, b: str, backend: str = "semantic") -> float:
    """Compute similarity between two strings using the selected backend.

    Returns cosine similarity in [0, 1].
    """
    import math

    vecs = embed_paragraphs([a, b], backend=backend)
    if hasattr(vecs, "shape") and getattr(vecs, "ndim", 2) >= 2:
        v0, v1 = vecs[0], vecs[1]
        # cosine
        dot = float(sum(x * y for x, y in zip(v0, v1)))
        norm0 = math.sqrt(sum(x * x for x in v0))
        norm1 = math.sqrt(sum(x * x for x in v1))
        if norm0 == 0 or norm1 == 0:
            return 0.0
        return max(0.0, min(1.0, dot / (norm0 * norm1)))
    return 0.0


def identify_unique_alpha_risks(
    target_paragraphs: List[str],
    peer_paragraphs_by_ticker: Dict[str, List[str]],
    similarity_backend: str = "semantic",
    unique_threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """Identify paragraphs in `target_paragraphs` that are unique vs peers.

    Returns a list of dicts with keys: `paragraph`, `max_similarities` (per-peer),
    `peer_group_stat` (median of maxima), and `is_unique` (bool).
    """
    from statistics import median

    results: List[Dict[str, Any]] = []

    # Precompute embeddings for peers per ticker
    peer_embeds: Dict[str, Any] = {}
    for tk, paras in peer_paragraphs_by_ticker.items():
        if not paras:
            peer_embeds[tk] = None
            continue
        peer_embeds[tk] = embed_paragraphs(paras, backend=similarity_backend)

    for p in target_paragraphs:
        per_peer_max: List[float] = []
        # compute similarity against each peer's paragraphs and take max
        for tk, paras in peer_paragraphs_by_ticker.items():
            if not paras:
                per_peer_max.append(0.0)
                continue
            # compute similarities to this peer's paragraphs
            sims = [compute_semantic_similarity(p, q, backend=similarity_backend) for q in paras]
            per_peer_max.append(max(sims) if sims else 0.0)

        pg_stat = median(per_peer_max) if per_peer_max else 0.0
        is_unique = pg_stat < unique_threshold
        results.append({"paragraph": p, "max_similarities": per_peer_max, "peer_group_stat": pg_stat, "is_unique": is_unique})

    return results
