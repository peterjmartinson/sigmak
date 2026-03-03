"""CLI handler for the `download` subcommand."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from sigmak.downloads.tenk_downloader import (
    TenKDownloader,
    fetch_company_submissions,
    resolve_ticker_to_cik,
)
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
