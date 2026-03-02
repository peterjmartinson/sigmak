"""
TDD tests for the yfinance peer discovery adapter.

Issue: GET_PEERS_WITH_YFINANCE (see documentation/improve-peer-selection/)

All tests are fully mocked — no real network calls.
Run with: uv run pytest tests/test_peer_discovery_yfinance.py -v
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, call, patch

import pytest

from sigmak.adapters.yfinance_adapter import PeerRecord, YFinanceAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(tmp_path: Path, **kwargs: Any) -> YFinanceAdapter:
    """Return a YFinanceAdapter backed by a temp directory."""
    defaults = dict(
        cache_dir=tmp_path,
        ttl_seconds=86400,
        rate_limit_rps=100.0,  # no throttle in tests
        max_retries=3,
        backoff_base=0.0,      # no sleep in tests
    )
    defaults.update(kwargs)
    return YFinanceAdapter(**defaults)


def _ticker_info(
    ticker: str,
    market_cap: int | None = 1_000_000_000,
    industry: str = "Software",
    sector: str = "Technology",
    exchange: str = "NMS",
    short_name: str | None = None,
) -> Dict[str, Any]:
    return {
        "symbol": ticker,
        "shortName": short_name or f"{ticker} Corp",
        "marketCap": market_cap,
        "exchange": exchange,
        "industry": industry,
        "sector": sector,
        "industryKey": industry.lower().replace(" ", "-"),
        "sectorKey": sector.lower().replace(" ", "-"),
    }


# ---------------------------------------------------------------------------
# Test 1 — bulk fetch calls yf.Tickers once and writes cache
# ---------------------------------------------------------------------------

def test_bulk_fetch_uses_single_yf_tickers_call_and_caches(tmp_path: Path) -> None:
    """
    SRP: Adapter must call yf.Tickers exactly once for a ticker list,
    normalise the result, and write a JSON cache file.
    """
    adapter = _make_adapter(tmp_path)

    mock_tickers_obj = MagicMock()
    mock_tickers_obj.tickers = {
        "AAPL": MagicMock(info=_ticker_info("AAPL", market_cap=3_000_000_000_000)),
        "MSFT": MagicMock(info=_ticker_info("MSFT", market_cap=2_500_000_000_000)),
    }

    with patch("sigmak.adapters.yfinance_adapter.yf") as mock_yf:
        mock_yf.Tickers.return_value = mock_tickers_obj
        result = adapter.fetch_bulk(["AAPL", "MSFT"])

    # yf.Tickers must be called exactly once
    mock_yf.Tickers.assert_called_once()

    assert "AAPL" in result
    assert "MSFT" in result
    assert result["AAPL"]["market_cap"] == 3_000_000_000_000
    assert result["MSFT"]["ticker"] == "MSFT"

    # Cache file should exist
    cache_files = list(tmp_path.glob("yfinance/*.json"))
    assert len(cache_files) == 1


def test_bulk_fetch_cache_hit_skips_network_call(tmp_path: Path) -> None:
    """
    SRP: A second call with the same tickers must read from cache,
    not call yf.Tickers again.
    """
    adapter = _make_adapter(tmp_path)

    mock_tickers_obj = MagicMock()
    mock_tickers_obj.tickers = {
        "AAPL": MagicMock(info=_ticker_info("AAPL")),
    }

    with patch("sigmak.adapters.yfinance_adapter.yf") as mock_yf:
        mock_yf.Tickers.return_value = mock_tickers_obj
        adapter.fetch_bulk(["AAPL"])  # first call — hits network
        adapter.fetch_bulk(["AAPL"])  # second call — should hit cache

    # Network should only have been hit once
    assert mock_yf.Tickers.call_count == 1


# ---------------------------------------------------------------------------
# Test 2 — market-cap fraction filter
# ---------------------------------------------------------------------------

def test_filter_by_marketcap_and_fraction(tmp_path: Path) -> None:
    """
    SRP: confirm peers with market_cap >= max(min_abs, fraction * target) are kept.
    Default fraction=0.10, min_abs=50_000_000.
    Target market_cap = 1_000_000_000 → threshold = 100_000_000.
    """
    adapter = _make_adapter(tmp_path)

    candidates = {
        "BIG":  {"ticker": "BIG",  "market_cap": 500_000_000, "company_name": "Big Co",  "exchange": "NYSE", "industry": "Software", "sector": "Technology"},
        "MID":  {"ticker": "MID",  "market_cap": 100_000_000, "company_name": "Mid Co",  "exchange": "NYSE", "industry": "Software", "sector": "Technology"},
        "TINY": {"ticker": "TINY", "market_cap":  10_000_000, "company_name": "Tiny Co", "exchange": "NYSE", "industry": "Software", "sector": "Technology"},
        "NONE": {"ticker": "NONE", "market_cap": None,        "company_name": "None Co", "exchange": "NYSE", "industry": "Software", "sector": "Technology"},
    }

    result = adapter._apply_filter_pipeline(candidates, target_market_cap=1_000_000_000)

    tickers = [p.ticker for p in result]
    assert "BIG" in tickers
    assert "MID" in tickers   # exactly at threshold — included
    assert "TINY" not in tickers
    # None market_cap excluded when target is known
    assert "NONE" not in tickers


# ---------------------------------------------------------------------------
# Test 3 — missing target market_cap -> percentile fallback
# ---------------------------------------------------------------------------

def test_missing_target_marketcap_percentile_fallback(tmp_path: Path) -> None:
    """
    SRP: When target market_cap is None, select peers by top-50th-percentile
    among candidates rather than crashing or returning nothing.
    """
    adapter = _make_adapter(tmp_path)

    candidates = {
        t: {"ticker": t, "market_cap": mc, "company_name": f"{t} Co",
            "exchange": "NYSE", "industry": "Software", "sector": "Technology"}
        for t, mc in [("A", 900_000_000), ("B", 700_000_000),
                      ("C", 500_000_000), ("D", 300_000_000),
                      ("E", 100_000_000)]
    }

    result = adapter._apply_filter_pipeline(candidates, target_market_cap=None)

    # Must return something (top half = A, B, C)
    assert len(result) >= 1
    tickers = [p.ticker for p in result]
    # Largest should always be included
    assert "A" in tickers


# ---------------------------------------------------------------------------
# Test 4 — min_peers relaxation
# ---------------------------------------------------------------------------

def test_min_peers_relaxation_progresses_correctly(tmp_path: Path) -> None:
    """
    SRP: If filtering at fraction=0.10 yields fewer than min_peers,
    the adapter relaxes to 0.05 then 0.02 until min_peers is met or exhausted.
    """
    adapter = _make_adapter(tmp_path, min_peers=3)

    # Only 2 candidates pass 0.10 threshold (target=10B → threshold=1B)
    # All 4 pass 0.02 threshold (threshold=200M)
    candidates = {
        t: {"ticker": t, "market_cap": mc, "company_name": f"{t} Co",
            "exchange": "NYSE", "industry": "Software", "sector": "Technology"}
        for t, mc in [
            ("A", 5_000_000_000),
            ("B", 2_000_000_000),
            ("C",   500_000_000),
            ("D",   300_000_000),
        ]
    }

    result = adapter._apply_filter_pipeline(candidates, target_market_cap=10_000_000_000)

    # With relaxation, should include at least min_peers (3)
    assert len(result) >= 3


# ---------------------------------------------------------------------------
# Test 5 — backoff + retries on transient errors
# ---------------------------------------------------------------------------

def test_backoff_and_retries_on_transient_errors(tmp_path: Path) -> None:
    """
    SRP: When yf.Tickers raises a transient error, the adapter retries
    up to max_retries times before giving up and returning {}.
    """
    adapter = _make_adapter(tmp_path, max_retries=3, backoff_base=0.0)

    with patch("sigmak.adapters.yfinance_adapter.yf") as mock_yf, \
         patch("sigmak.adapters.yfinance_adapter.time") as mock_time:
        mock_yf.Tickers.side_effect = ConnectionError("network blip")
        mock_time.sleep = MagicMock()
        result = adapter.fetch_bulk(["AAPL"])

    # Should have retried max_retries times
    assert mock_yf.Tickers.call_count == adapter.max_retries
    # Should return empty dict (not crash)
    assert result == {}
    # Should have slept between retries (backoff_base=0 → sleep(0) still called)
    assert mock_time.sleep.call_count == adapter.max_retries - 1


# ---------------------------------------------------------------------------
# Test 6 — yf.Industry probed defensively
# ---------------------------------------------------------------------------

def test_industry_object_probed_defensively(tmp_path: Path) -> None:
    """
    SRP: _get_industry_candidates() must try top_companies, then tickers, then
    members. If none exist, return [] without raising.
    """
    adapter = _make_adapter(tmp_path)

    # Object with none of the expected attributes
    bare_mock = MagicMock(spec=[])  # spec=[] means no attributes

    with patch("sigmak.adapters.yfinance_adapter.yf") as mock_yf:
        mock_yf.Industry.return_value = bare_mock
        result = adapter._get_industry_candidates("software-application")

    assert result == []


def test_industry_object_top_companies_dataframe(tmp_path: Path) -> None:
    """
    SRP: When yf.Industry has .top_companies (a DataFrame), use its index as tickers.
    """
    import pandas as pd

    adapter = _make_adapter(tmp_path)

    mock_industry = MagicMock()
    mock_industry.top_companies = pd.DataFrame(
        {"name": ["Apple Inc", "Microsoft Corp"]},
        index=["AAPL", "MSFT"]
    )

    with patch("sigmak.adapters.yfinance_adapter.yf") as mock_yf:
        mock_yf.Industry.return_value = mock_industry
        result = adapter._get_industry_candidates("software-application")

    assert "AAPL" in result
    assert "MSFT" in result


# ---------------------------------------------------------------------------
# Test 7 — adapter disabled by default / env guard
# ---------------------------------------------------------------------------

def test_adapter_disabled_returns_empty_list(tmp_path: Path) -> None:
    """
    SRP: get_peers_via_yfinance on PeerDiscoveryService returns [] and does
    not call the adapter when SIGMAK_PEER_YFINANCE_ENABLED is not 'true'.
    """
    from sigmak.peer_discovery import PeerDiscoveryService

    svc = PeerDiscoveryService(cache_dir=tmp_path)

    with patch.dict("os.environ", {"SIGMAK_PEER_YFINANCE_ENABLED": "false"}):
        with patch("sigmak.adapters.yfinance_adapter.yf") as mock_yf:
            result = svc.get_peers_via_yfinance("MSFT", n=5)

    assert result == []
    mock_yf.Tickers.assert_not_called()


def test_adapter_enabled_calls_get_peers(tmp_path: Path) -> None:
    """
    SRP: When SIGMAK_PEER_YFINANCE_ENABLED=true, get_peers_via_yfinance
    delegates to YFinanceAdapter.get_peers() and returns the result.
    """
    from sigmak.peer_discovery import PeerDiscoveryService

    expected = [PeerRecord(
        ticker="MSFT", company_name="Microsoft Corp",
        market_cap=2_500_000_000_000, exchange="NMS",
        industry="Software", sector="Technology",
        source="yfinance", enriched_at="2026-02-20T00:00:00Z",
    )]

    svc = PeerDiscoveryService(cache_dir=tmp_path)

    with patch.dict("os.environ", {"SIGMAK_PEER_YFINANCE_ENABLED": "true"}):
        with patch("sigmak.adapters.yfinance_adapter.YFinanceAdapter.get_peers", return_value=expected):
            result = svc.get_peers_via_yfinance("MSFT", n=5)

    assert result == expected
