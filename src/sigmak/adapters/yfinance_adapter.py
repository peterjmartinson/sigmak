"""
Optional yfinance-based peer discovery adapter.

This adapter is OPT-IN only (disabled by default).
Enable via: SIGMAK_PEER_YFINANCE_ENABLED=true

Design:
- PeerRecord   : lightweight @dataclass for a single enriched peer
- YFinanceAdapter : fetches, caches, filters, and ranks peer candidates
  1. Resolve target ticker's industryKey via yf.Ticker(target).info
  2. Probe yf.Industry(industryKey) defensively for a top-company list
  3. Bulk-enrich candidates with yf.Tickers
  4. Apply market-cap believability filter with progressive relaxation
  5. Upsert enriched peers into the filings DB as a side-effect
  6. Return List[PeerRecord] sorted by market_cap DESC, ticker ASC

NOTE: yfinance wraps unofficial Yahoo Finance endpoints. This feature is
intentionally opt-in for demo/PoC usage. Consult Yahoo's Terms of Service
before using in sustained production workflows.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time as time  # aliased so tests can patch 'sigmak.adapters.yfinance_adapter.time'
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yfinance as yf  # noqa: F401
    import pandas as pd    # noqa: F401
except ImportError:  # pragma: no cover
    yf = None
    pd = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------

@dataclass
class PeerRecord:
    """Enriched peer metadata returned by YFinanceAdapter.get_peers()."""

    ticker: str
    company_name: Optional[str]
    market_cap: Optional[int]
    exchange: Optional[str]
    industry: Optional[str]
    sector: Optional[str]
    source: str
    enriched_at: str  # ISO-8601 UTC


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class YFinanceAdapter:
    """
    Fetch, cache, and rank peer candidates using yfinance.

    Parameters
    ----------
    cache_dir        : Base cache directory. Payloads written to cache_dir/yfinance/
    ttl_seconds      : Cache time-to-live in seconds (default 24 h).
    rate_limit_rps   : Soft rate limit (requests per second) — currently applied
                       as a minimum sleep between retries only (yf.Tickers is one
                       bulk request regardless of ticker count).
    max_retries      : Number of attempts for transient network errors.
    backoff_base     : Base seconds for exponential backoff between retries.
    min_peers        : Minimum peers to return; triggers threshold relaxation.
    min_fraction     : Initial market-cap fraction threshold (default 0.10).
    min_abs_cap      : Absolute minimum market_cap floor in USD (default $50 M).
    n_peers          : Maximum peers to return (default 10).
    """

    _FRACTION_STEPS: List[float] = [0.10, 0.05, 0.02, 0.0]

    def __init__(
        self,
        cache_dir: Path,
        ttl_seconds: int = 86400,
        rate_limit_rps: float = 1.0,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        min_peers: int = 5,
        min_fraction: float = 0.10,
        min_abs_cap: int = 50_000_000,
        n_peers: int = 10,
    ) -> None:
        self.cache_dir = Path(cache_dir) / "yfinance"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self.rate_limit_rps = rate_limit_rps
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.min_peers = min_peers
        self.min_fraction = min_fraction
        self.min_abs_cap = min_abs_cap
        self.n_peers = n_peers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_peers(self, ticker: str, n: Optional[int] = None, db_path: Optional[str] = None) -> List[PeerRecord]:
        """
        Main entry point: return up to n enriched PeerRecords for ticker.

        Side-effect: upserts market_cap / exchange / industry back into
        the filings DB (peers table) via upsert_peer().
        """
        n = n or self.n_peers
        ticker = ticker.upper()

        # 1. Resolve industry key for target
        industry_key = self._resolve_industry_key(ticker)
        if not industry_key:
            logger.warning("yfinance: could not resolve industryKey for %s", ticker)
            return []

        # 2. Get candidate tickers from industry object
        candidate_tickers = self._get_industry_candidates(industry_key)
        # Always include the target itself for enrichment (skip it in output)
        all_tickers = list({ticker} | set(candidate_tickers))

        if not all_tickers:
            logger.warning("yfinance: no candidate tickers found for industry %s", industry_key)
            return []

        # 3. Bulk-enrich
        enriched = self.fetch_bulk(all_tickers)

        target_data = enriched.get(ticker, {})
        target_market_cap: Optional[int] = target_data.get("market_cap")

        # Exclude target from peer candidates
        candidates = {t: d for t, d in enriched.items() if t != ticker}

        # 4. Apply filter + sort
        peers = self._apply_filter_pipeline(candidates, target_market_cap)[:n]

        # 5. Upsert back into DB (best-effort)
        if db_path:
            self._upsert_peers_to_db(peers, db_path)

        return peers

    # ------------------------------------------------------------------
    # Industry candidate discovery
    # ------------------------------------------------------------------

    def _resolve_industry_key(self, ticker: str) -> Optional[str]:
        """Get Yahoo industryKey for a ticker via yf.Ticker.info."""
        if yf is None:  # pragma: no cover
            logger.error("yfinance not installed")
            return None
        try:
            info = yf.Ticker(ticker).info
            raw: Any = info.get("industryKey") or info.get("industry")
            return str(raw) if raw is not None else None
        except Exception as exc:
            logger.warning("yfinance: failed to resolve industryKey for %s: %s", ticker, exc)
            return None

    def _get_industry_candidates(self, industry_key: str) -> List[str]:
        """
        Probe yf.Industry(industry_key) defensively for member tickers.

        Tries in order:
          1. .top_companies  (DataFrame — use index as tickers)
          2. .tickers        (list or dict)
          3. .members        (list or dict)

        Returns [] if none succeed.
        """
        if yf is None:  # pragma: no cover
            return []
        try:
            industry_obj = yf.Industry(industry_key)
        except Exception as exc:
            logger.warning("yfinance: yf.Industry(%s) raised: %s", industry_key, exc)
            return []

        # --- try top_companies (DataFrame) ---
        top = getattr(industry_obj, "top_companies", None)
        if top is not None:
            try:
                if pd is not None and isinstance(top, pd.DataFrame):
                    tickers = [str(t).upper() for t in top.index.tolist()]
                    logger.debug("yfinance: industry %s → top_companies → %d tickers", industry_key, len(tickers))
                    return tickers
            except Exception as exc:
                logger.debug("yfinance: top_companies probe failed: %s", exc)

        # --- try .tickers ---
        tickers_attr = getattr(industry_obj, "tickers", None)
        if tickers_attr is not None:
            try:
                if isinstance(tickers_attr, dict):
                    tickers = [str(t).upper() for t in tickers_attr.keys()]
                else:
                    tickers = [str(t).upper() for t in tickers_attr]
                logger.debug("yfinance: industry %s → .tickers → %d tickers", industry_key, len(tickers))
                return tickers
            except Exception as exc:
                logger.debug("yfinance: .tickers probe failed: %s", exc)

        # --- try .members ---
        members_attr = getattr(industry_obj, "members", None)
        if members_attr is not None:
            try:
                if isinstance(members_attr, dict):
                    tickers = [str(t).upper() for t in members_attr.keys()]
                else:
                    tickers = [str(t).upper() for t in members_attr]
                logger.debug("yfinance: industry %s → .members → %d tickers", industry_key, len(tickers))
                return tickers
            except Exception as exc:
                logger.debug("yfinance: .members probe failed: %s", exc)

        logger.warning(
            "yfinance: yf.Industry(%s) has no usable top_companies/tickers/members attribute. "
            "Returning empty candidate list.",
            industry_key,
        )
        return []

    # ------------------------------------------------------------------
    # Bulk enrichment with caching
    # ------------------------------------------------------------------

    def _cache_key(self, tickers: List[str]) -> str:
        joined = ",".join(sorted(t.upper() for t in tickers))
        return hashlib.md5(joined.encode()).hexdigest()

    def _cache_path(self, tickers: List[str]) -> Path:
        return self.cache_dir / f"{self._cache_key(tickers)}.json"

    def _is_cache_valid(self, path: Path) -> bool:
        if not path.exists():
            return False
        age = time.time() - path.stat().st_mtime
        return age < self.ttl_seconds

    def fetch_bulk(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch yfinance metadata for a list of tickers.

        - Returns from disk cache if TTL has not elapsed.
        - Uses yf.Tickers (single bulk request) for all tickers.
        - Retries up to max_retries on transient errors with exponential backoff.
        - Returns {} if all retries fail.
        """
        if not tickers:
            return {}

        cache_file = self._cache_path(tickers)

        # Cache hit
        if self._is_cache_valid(cache_file):
            logger.debug("yfinance: cache hit for %d tickers → %s", len(tickers), cache_file)
            try:
                cached: Dict[str, Dict[str, Any]] = json.loads(cache_file.read_text(encoding="utf-8"))
                return cached
            except Exception as exc:
                logger.warning("yfinance: corrupt cache %s: %s — refetching", cache_file, exc)

        # Cache miss → fetch with retries
        joined = " ".join(t.upper() for t in tickers)
        raw_data: Dict[str, Any] = {}

        for attempt in range(self.max_retries):
            try:
                if yf is None:  # pragma: no cover
                    raise RuntimeError("yfinance not installed")
                tickers_obj = yf.Tickers(joined)
                raw_data = tickers_obj.tickers
                logger.info("yfinance: fetched %d tickers (attempt %d)", len(tickers), attempt + 1)
                break
            except Exception as exc:
                logger.warning("yfinance: fetch attempt %d/%d failed: %s", attempt + 1, self.max_retries, exc)
                if attempt < self.max_retries - 1:
                    sleep_secs = self.backoff_base * (2 ** attempt)
                    time.sleep(sleep_secs)
        else:
            logger.error("yfinance: all %d retries exhausted — returning {}", self.max_retries)
            return {}

        # Normalize
        enriched_at = datetime.now(timezone.utc).isoformat()
        normalized: Dict[str, Dict[str, Any]] = {}
        for t, ticker_obj in raw_data.items():
            t_up = t.upper()
            try:
                info: Dict[str, Any] = ticker_obj.info or {}
            except Exception:
                info = {}
            normalized[t_up] = {
                "ticker": t_up,
                "company_name": info.get("shortName") or info.get("longName") or info.get("companyName"),
                "market_cap": info.get("marketCap"),
                "exchange": info.get("exchange"),
                "industry": info.get("industry"),
                "sector": info.get("sector"),
                "industry_key": info.get("industryKey"),
                "sector_key": info.get("sectorKey"),
                "enriched_at": enriched_at,
            }

        # Write cache
        try:
            cache_file.write_text(json.dumps(normalized, default=str), encoding="utf-8")
            logger.debug("yfinance: wrote cache %s", cache_file)
        except Exception as exc:
            logger.warning("yfinance: could not write cache: %s", exc)

        return normalized

    # ------------------------------------------------------------------
    # Filter pipeline
    # ------------------------------------------------------------------

    def _apply_filter_pipeline(
        self,
        candidates: Dict[str, Dict[str, Any]],
        target_market_cap: Optional[int],
    ) -> List[PeerRecord]:
        """
        Apply believability filter with progressive relaxation.

        If target_market_cap is known:
          threshold = max(min_abs_cap, fraction * target_market_cap)
          Steps: 0.10 → 0.05 → 0.02 → 0.0 (include all with market_cap)

        If target_market_cap is None:
          Select peers in the top 50th percentile by market_cap.
        """
        enriched_at = datetime.now(timezone.utc).isoformat()

        def to_record(d: Dict[str, Any]) -> PeerRecord:
            return PeerRecord(
                ticker=d["ticker"].upper(),
                company_name=d.get("company_name"),
                market_cap=d.get("market_cap"),
                exchange=d.get("exchange"),
                industry=d.get("industry"),
                sector=d.get("sector"),
                source="yfinance",
                enriched_at=d.get("enriched_at") or enriched_at,
            )

        if target_market_cap is None:
            # Percentile fallback: top 50th percentile by market_cap
            with_cap = [(t, d) for t, d in candidates.items() if d.get("market_cap") is not None]
            if not with_cap:
                # Return all sorted by ticker if nothing has a cap
                return sorted([to_record(d) for d in candidates.values()], key=lambda p: p.ticker)
            caps = sorted([d["market_cap"] for _, d in with_cap], reverse=True)
            median_cap = caps[len(caps) // 2]
            filtered = [d for _, d in with_cap if d["market_cap"] >= median_cap]
            return self._sort_records([to_record(d) for d in filtered])

        # Known target market_cap — progressive threshold relaxation
        for fraction in self._FRACTION_STEPS:
            threshold = max(self.min_abs_cap, fraction * target_market_cap)
            filtered = [
                d for d in candidates.values()
                if d.get("market_cap") is not None and d["market_cap"] >= threshold
            ]
            if len(filtered) >= self.min_peers or fraction == 0.0:
                return self._sort_records([to_record(d) for d in filtered])

        return []

    @staticmethod
    def _sort_records(records: List[PeerRecord]) -> List[PeerRecord]:
        """Sort by market_cap DESC (NULLs last), then ticker ASC for stability."""
        return sorted(
            records,
            key=lambda p: (-(p.market_cap or 0), p.ticker),
        )

    # ------------------------------------------------------------------
    # DB side-effect
    # ------------------------------------------------------------------

    def _upsert_peers_to_db(self, peers: List[PeerRecord], db_path: str) -> None:
        """Write enriched market_cap / exchange / industry back into the peers table."""
        try:
            from sigmak.filings_db import upsert_peer
        except ImportError:  # pragma: no cover
            logger.warning("yfinance: sigmak.filings_db not available; skipping DB upsert")
            return

        for peer in peers:
            try:
                upsert_peer(
                    db_path,
                    peer.ticker,
                    cik=None,
                    sic=None,
                    industry=peer.industry,
                    market_cap=float(peer.market_cap) if peer.market_cap is not None else None,
                )
            except Exception as exc:
                logger.warning("yfinance: upsert_peer failed for %s: %s", peer.ticker, exc)
