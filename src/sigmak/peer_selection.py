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

# A small hard-coded fallback 'Magnificent 7' list (used only as a fallback)
MAGNIFICENT_7 = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]

# A broader seed pool of large-cap tickers to use for dynamic discovery when
# yfinance cannot enumerate an entire sector. This is intentionally small and
# conservative to avoid heavy network use.
LARGE_TICKER_POOL = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "NVDA",
    "TSLA",
    "INTC",
    "ORCL",
    "IBM",
    "CSCO",
    "ADBE",
    "CRM",
    "AVGO",
    "QCOM",
]

# Seed dictionary for top ~50 global companies (approximate marketCap used
# as a fallback when yfinance returns missing metadata). Values are in USD.
TOP50_SEED: Dict[str, int] = {
    "AAPL": 2500000000000,
    "MSFT": 2300000000000,
    "AMZN": 1400000000000,
    "GOOGL": 1700000000000,
    "META": 900000000000,
    "NVDA": 1000000000000,
    "TSLA": 800000000000,
    "BRK.A": 750000000000,
    "TSM": 600000000000,
    "V": 500000000000,
    "JPM": 450000000000,
    "JNJ": 420000000000,
    "WMT": 400000000000,
    "BABA": 200000000000,
    "DIS": 200000000000,
    "NVDAOLD": 100000000000,
}


class TargetMetadataError(RuntimeError):
    """Raised when target market metadata cannot be determined safely."""



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


def discover_peers(ticker: str) -> List[str]:
    """Discover large peers for `ticker` by matching sector/industry from a
    conservative seed pool. Returns a list of tickers ordered by marketCap.

    This is intentionally lightweight: it only probes a small, curated pool
    of large tickers to avoid broad network scans.
    """
    yf = _try_import_yfinance()
    if not yf:
        logger.debug("yfinance not available for discover_peers")
        return []

    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        target_sector = info.get("sector")
        target_industry = info.get("industry")
    except Exception:
        logger.warning("discover_peers: failed to fetch info for %s", ticker)
        return []

    candidates: List[Tuple[str, Optional[float], Optional[str], Optional[str]]] = []
    for t in LARGE_TICKER_POOL:
        try:
            info = yf.Ticker(t).info or {}
            cap = info.get("marketCap")
            sector = info.get("sector")
            industry = info.get("industry")
            # prefer same sector or same industry
            if sector == target_sector or (target_industry and industry == target_industry):
                candidates.append((t, cap, sector, industry))
        except Exception:
            continue

    # Sort by market cap descending and return tickers
    candidates = [c for c in candidates if c[1]]
    candidates.sort(key=lambda x: x[1] or 0, reverse=True)
    return [c[0] for c in candidates]


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
    # If no marketCap available for target, try the TOP50 seed dictionary
    if not target_cap:
        seed_cap = TOP50_SEED.get(target_ticker.upper())
        if seed_cap:
            target_cap = seed_cap
            market_caps[target_ticker] = seed_cap
            logger.info("Using seeded marketCap for %s (%d)", target_ticker, seed_cap)
        else:
            # If we don't have a seeded value, raise to avoid producing
            # misleading peer groups for major tickers.
            raise TargetMetadataError(f"Missing marketCap for target {target_ticker}; cannot validate peers safely")

    # Detect mega-cap (special handling). New threshold: Mega-cap >= $200B
    MEGA_CAP = 200_000_000_000

    # If target is Mega-cap, try dynamic discovery and size-compatible filtering
    if target_cap >= MEGA_CAP:
        # Dynamic discovery by sector/industry using a small pool + yfinance
        try:
            discovered = discover_peers(target_ticker)
        except Exception:
            discovered = []

        # Candidate pool: discovered first, otherwise fallback to peer_tickers
        candidates = discovered or peer_tickers[:]

        # Size-compatibility filter per spec: 0.5x - 5.0x
        low, high = 0.5 * target_cap, 5.0 * target_cap
        # Additionally, exclude any peers below an absolute floor of $10B
        ABS_MIN = 10_000_000_000
        cand_with_caps = [(t, market_caps.get(t) or 0) for t in candidates]
        cand_filtered = [ (t, cap) for t, cap in cand_with_caps if cap and low <= cap <= high and cap >= ABS_MIN ]

        # If none matched, try the hard-coded Magnificent 7 as a final fallback
        if not cand_filtered:
            cand_with_caps = [(t, market_caps.get(t) or TOP50_SEED.get(t) or 0) for t in MAGNIFICENT_7]
            cand_filtered = [ (t, cap) for t, cap in cand_with_caps if cap and low <= cap <= high and cap >= ABS_MIN ]

        # Score by absolute marketCap difference and return the 6 closest
        cand_filtered.sort(key=lambda x: abs((x[1] or 0) - target_cap))
        return [t for t, _ in cand_filtered][:6]

    # Non-mega default behavior: keep the previous order-of-magnitude rule
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

    # If the target is a Mega-cap, prefer only Mega-cap peers in this fallback
    # to avoid skewing comparisons (e.g., AAPL vs tiny caps). Mega-cap defined
    # as marketCap >= 100 billion.
    MEGA_CAP = 100_000_000_000
    try:
        if target_cap and target_cap >= MEGA_CAP:
            mega_peers = [t for t in ranked if (market_caps.get(t) or 0) >= MEGA_CAP]
            if mega_peers:
                return mega_peers[:10]
    except Exception:
        # on any error, fall back to usual ranked list
        pass

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
    prev_paragraphs: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """Identify paragraphs in `target_paragraphs` that are unique vs peers.

    Returns a list of dicts with keys: `paragraph`, `max_similarities` (per-peer),
    `peer_group_stat` (median of maxima), and `is_unique` (bool).
    """
    from statistics import median

    results: List[Dict[str, Any]] = []

    # Precompute embeddings for target and peers to avoid redundant encoding
    # Embed target paragraphs
    if target_paragraphs:
        target_emb = embed_paragraphs(target_paragraphs, backend=similarity_backend)
    else:
        target_emb = []

    peer_embeds: Dict[str, Any] = {}
    for tk, paras in peer_paragraphs_by_ticker.items():
        if not paras:
            peer_embeds[tk] = None
            continue
        peer_embeds[tk] = embed_paragraphs(paras, backend=similarity_backend)

    # Precompute embeddings for previous-year paragraphs if provided
    prev_emb = None
    if prev_paragraphs:
        prev_emb = embed_paragraphs(prev_paragraphs, backend=similarity_backend)

    # helper: normalize vectors (works for nested lists or numpy arrays)
    def _l2_normalize(mat: Any) -> List[List[float]]:
        try:
            import numpy as _np

            arr = _np.array(mat, dtype=float)
            norms = _np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return (arr / norms).tolist()
        except Exception:
            # pure-python fallback
            normed: List[List[float]] = []
            for row in list(mat):
                import math

                norm = math.sqrt(sum(float(x) * float(x) for x in row))
                if norm == 0:
                    norm = 1.0
                normed.append([float(x) / norm for x in row])
            return normed

    # Normalize all embeddings where present
    t_emb_norm = _l2_normalize(target_emb) if hasattr(target_emb, "__len__") and len(target_emb) else []
    peer_emb_norm: Dict[str, List[List[float]]] = {}
    for tk, mat in peer_embeds.items():
        if mat is None:
            peer_emb_norm[tk] = []
        else:
            peer_emb_norm[tk] = _l2_normalize(mat)

    prev_emb_norm = _l2_normalize(prev_emb) if prev_emb is not None else None

    # compute per-paragraph stats
    from statistics import median

    for idx, p in enumerate(target_paragraphs):
        per_peer_max: List[float] = []
        v0 = t_emb_norm[idx] if idx < len(t_emb_norm) else None
        for tk, paras in peer_paragraphs_by_ticker.items():
            mats = peer_emb_norm.get(tk, [])
            if not mats or v0 is None:
                per_peer_max.append(0.0)
                continue
            # compute max cosine similarity between v0 and mats
            best = 0.0
            for v in mats:
                dot = sum(float(a) * float(b) for a, b in zip(v0, v))
                if dot > best:
                    best = dot
            per_peer_max.append(best)

        pg_stat = median(per_peer_max) if per_peer_max else 0.0
        is_unique = pg_stat < unique_threshold
        results.append({"paragraph": p, "max_similarities": per_peer_max, "peer_group_stat": pg_stat, "is_unique": is_unique})

    # substantive_change_score: percent of target paragraphs that have NO
    # semantic match (>0.85) in prev_paragraphs (if provided).
    substantive_change_score = 0.0
    if prev_emb_norm is not None and t_emb_norm:
        no_match = 0
        for v0 in t_emb_norm:
            best = 0.0
            for v in prev_emb_norm:
                dot = sum(float(a) * float(b) for a, b in zip(v0, v))
                if dot > best:
                    best = dot
            if best <= 0.85:
                no_match += 1
        substantive_change_score = (no_match / len(t_emb_norm)) * 100.0

    return results, substantive_change_score
