#!/usr/bin/env python3
"""Diagnostic for peer selection.

Prints:
- initial candidate list from PeerDiscoveryService
- validated list from peer_selection.validate_peer_group
- presence/absence of local filing HTML for first N candidates (no downloads)

Usage: python3 scripts/diagnose_peer_selection.py TICKER YEAR [N]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, "src")

from sigmak.peer_discovery import PeerDiscoveryService
from sigmak import peer_selection


def find_html_in_dir_simple(ticker_dir: Path, year: int) -> Path | None:
    if not ticker_dir.exists():
        return None
    # exact year folder
    exact = ticker_dir / str(year)
    if exact.exists() and any(exact.glob("*.htm")):
        return next(exact.glob("*.htm"))
    # any html under ticker_dir that mentions the year in filename or parent
    candidates = list(ticker_dir.rglob("*.htm")) + list(ticker_dir.rglob("*.html"))
    for c in candidates:
        if str(year) in c.name or str(year) in str(c.parent.name):
            return c
    return None


def main():
    if len(sys.argv) < 3:
        print("Usage: diagnose_peer_selection.py TICKER YEAR [N]")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    year = int(sys.argv[2])
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    svc = PeerDiscoveryService(db_path="./database/sec_filings.db")
    print(f"Requesting candidate peers for {ticker}...")
    candidates = svc.find_peers_for_ticker(ticker, top_n=30)
    print(f"Initial candidates ({len(candidates)}): {candidates[:30]}")

    print("Validating peer group using peer_selection.validate_peer_group()...")
    try:
        validated = peer_selection.validate_peer_group(ticker, [c.upper() for c in candidates])
    except Exception as e:
        validated = []
        print("validate_peer_group failed:", e)
    print(f"Validated peers ({len(validated)}): {validated}")

    print(f"Checking local filing HTML presence for first {N} candidates (no downloads):")
    base = Path("data/filings")
    for cand in candidates[:N]:
        td = base / cand.upper()
        html = find_html_in_dir_simple(td, year)
        status = "FOUND" if html else "MISSING"
        print(f" - {cand.upper():6} : {status}{' -> ' + str(html) if html else ''}")


if __name__ == "__main__":
    main()
