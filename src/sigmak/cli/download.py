"""CLI handler for the `download` subcommand."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from sigmak.downloads.tenk_downloader import (
    TenKDownloader,
    fetch_company_submissions,
    resolve_ticker_to_cik,
)
from sigmak.filings_db import get_peer, get_peers_by_sic
from sigmak.peer_discovery import PeerDiscoveryService

logger = logging.getLogger(__name__)


def _download_one(
    downloader: TenKDownloader,
    ticker: str,
    year: Optional[int],
) -> Tuple[str, str]:
    """Download one 10-K filing; returns (ticker, human-readable message)."""
    year_label = str(year) if year is not None else "latest"
    prefix = f"Checking for {year_label}:"

    try:
        cik = resolve_ticker_to_cik(ticker)
    except ValueError as e:
        logger.warning("Could not resolve %s: %s", ticker, e)
        return ticker, f"{prefix} Unknown ticker — could not resolve CIK"

    try:
        filings = fetch_company_submissions(cik, form_type="10-K")
    except Exception as e:
        logger.warning("Failed to fetch submissions for %s: %s", ticker, e)
        return ticker, f"{prefix} EDGAR query failed ({e})"

    if not filings:
        return ticker, f"{prefix} No filings found on EDGAR"

    chosen = None
    if year is not None:
        for f in filings:
            if str(year) == f.filing_date.split("-")[0]:
                chosen = f
                break

    exact_match = chosen is not None
    if chosen is None:
        chosen = sorted(filings, key=lambda x: x.filing_date, reverse=True)[0]

    actual_year = chosen.filing_date.split("-")[0]

    try:
        filing_id = downloader.db.insert_filing(chosen)
        existing = downloader.db.get_downloads_for_filing(filing_id)
        if existing:
            logger.info("Skipping existing download for %s %s", ticker, chosen.filing_date)
            return ticker, f"{prefix} {year_label} filing already downloaded. Skipping"
        downloader.download_filing(chosen, filing_id)
    except Exception as e:
        logger.exception("Download failed for %s", ticker)
        return ticker, f"{prefix} Download failed ({e})"

    if exact_match or year is None:
        return ticker, f"{prefix} {year_label} filing downloaded"
    else:
        return ticker, f"{prefix} {year_label} filing not found, most recent filing {actual_year} downloaded"


def run(
    ticker: str,
    years: List[int] | None,
    include_peers: bool,
    db_only: bool,
    max_peers: int = 6,
    explicit_peers: List[str] | None = None,
    db_path: str = "./database/sec_filings.db",
    download_dir: str = "./data/filings",
    **_: object,
) -> None:
    """Download 10-K filings for ticker and, optionally, its peers.

    Parameters
    ----------
    ticker:        Target company ticker symbol.
    years:         Filing years to download; ``None`` downloads the most recent
                   available 10-K for each company.
    include_peers: When True, auto-discover peers via yfinance and download
                   their filings alongside the target.
    db_only:       Unused here; present for kwarg-passthrough compatibility.
    max_peers:     Maximum number of peers to include when ``include_peers=True``.
    explicit_peers:
                   If provided, use as the peer list instead of auto-discovery.
    db_path:       Path to the SQLite filings database.
    download_dir:  Base directory where 10-K HTML files are stored.
    """
    downloader = TenKDownloader(db_path=db_path, download_dir=download_dir)
    svc = PeerDiscoveryService(db_path=db_path)

    target = ticker.upper()

    if explicit_peers is not None:
        peers = [p.upper() for p in explicit_peers if p.upper() != target]
    elif include_peers:
        peer_records = svc.get_peers_via_yfinance(target, n=max_peers)
        peers = [pr.ticker.upper() for pr in peer_records]
    else:
        peers = []

    to_process = [target] + peers

    year_list: List[Optional[int]] = list(years) if years else [None]

    summary: Dict[str, str] = {}
    for year in year_list:
        for t in to_process:
            t_key, message = _download_one(downloader, t, year)
            summary[t_key] = message

    print("\nDownload Summary:")
    for t, message in summary.items():
        print(f"  {t}: {message}")


def select_peers_strict_sic(
    svc: Any,
    db_path: str,
    target_ticker: str,
    year: int,
    max_peers: int = 6,
    require_filing_year: bool = False,
) -> List[str]:
    """Select peer tickers from the same SIC code, sorted by market-cap proximity.

    Parameters
    ----------
    svc:                PeerDiscoveryService (reserved for future use).
    db_path:            Path to the SQLite filings database.
    target_ticker:      Ticker symbol for the target company.
    year:               Filing year used when ``require_filing_year=True``.
    max_peers:          Maximum number of peers to return.
    require_filing_year:
                        When True, only return peers that have a 10-K filing
                        recorded for ``year``.

    Returns
    -------
    List of peer ticker strings ordered by ascending market-cap distance from
    the target.
    """
    target = get_peer(db_path, target_ticker)
    if target is None:
        return []

    target_cap: float = float(target.get("market_cap") or 0)
    sic: str = str(target.get("sic") or "")

    candidates: List[Dict[str, Any]] = get_peers_by_sic(db_path, sic, limit=500)

    filtered: List[Dict[str, Any]] = []
    for c in candidates:
        if c.get("ticker") == target_ticker:
            continue
        if require_filing_year:
            date_str = c.get("latest_10k_date") or ""
            if not date_str.startswith(str(year)):
                continue
        filtered.append(c)

    filtered.sort(key=lambda c: abs(float(c.get("market_cap") or 0) - target_cap))
    return [str(c["ticker"]) for c in filtered[:max_peers]]


def download_for_ticker(
    downloader: TenKDownloader,
    ticker: str,
    year: int,
    force_refresh: bool = False,
) -> Tuple[str, str]:
    """Download the 10-K for a single ticker and year.

    Parameters
    ----------
    downloader:    Configured :class:`~sigmak.downloads.tenk_downloader.TenKDownloader`.
    ticker:        Ticker symbol to download.
    year:          Target filing year.
    force_refresh: When True, re-download even if a record already exists.

    Returns
    -------
    ``(ticker, status)`` where *status* is one of:
    ``"downloaded"``       — exact year match downloaded successfully,
    ``"fallback_recent"``  — no exact year match; most recent filing downloaded,
    ``"already_exists"``   — filing already present and ``force_refresh`` is False,
    ``"no_filings"``       — EDGAR returned no filings for this ticker,
    ``"error"``            — an exception was raised during download.
    """
    try:
        cik = resolve_ticker_to_cik(ticker)
        filings = fetch_company_submissions(cik, form_type="10-K")
    except Exception as exc:
        logger.warning("download_for_ticker: failed to resolve/fetch %s: %s", ticker, exc)
        return ticker, "error"

    if not filings:
        return ticker, "no_filings"

    chosen = None
    for f in filings:
        if f.filing_date.startswith(str(year)):
            chosen = f
            break

    exact_match = chosen is not None
    if chosen is None:
        chosen = sorted(filings, key=lambda x: x.filing_date, reverse=True)[0]

    try:
        filing_id = downloader.db.insert_filing(chosen)
        if not force_refresh:
            existing = downloader.db.get_downloads_for_filing(filing_id)
            if existing:
                return ticker, "already_exists"
        downloader.download_filing(chosen, filing_id)
    except Exception as exc:
        logger.exception("download_for_ticker: download failed for %s: %s", ticker, exc)
        return ticker, "error"

    return ticker, "downloaded" if exact_match else "fallback_recent"
