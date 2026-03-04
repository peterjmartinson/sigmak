#!/usr/bin/env python3
# DEPRECATED: use 'uv run sigmak download --ticker TICKER' instead
"""Download 10-Ks for a target ticker and its industry peers.

This script now delegates to ``sigmak.cli.download.run``.

Preferred usage:
    uv run sigmak download --ticker NVDA

Legacy usage (still supported for backward compatibility):
    python scripts/download_peers_and_target.py NVDA 2024
    python scripts/download_peers_and_target.py NVDA 2024 --max-peers 6 --require-filing-year
"""
from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download 10-K for a target and its peers")
    parser.add_argument("ticker", help="Target ticker (e.g., NVDA)")
    parser.add_argument("--year", type=int, default=None, help="Filing year (e.g., 2024). If omitted, downloads the latest available 10-K.")
    parser.add_argument("--peers", nargs="+", help="Explicit peer tickers (optional)")
    parser.add_argument("--max-peers", type=int, default=6, help="Max peers to download (default: 6)")
    parser.add_argument("--require-filing-year", action="store_true", help="Require peer to have a filing for the given year")
    parser.add_argument("--force-refresh", action="store_true", help="Force re-download even if present")
    parser.add_argument("--db-path", type=str, default="./database/sec_filings.db", help="Path to filings DB")
    parser.add_argument("--download-dir", type=str, default="./data/filings", help="Base download dir")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    from sigmak.cli.download import run

    run(
        ticker=args.ticker.upper(),
        years=[args.year] if args.year else None,
        include_peers=args.peers is None,
        db_only=False,
        max_peers=args.max_peers,
        explicit_peers=[p.upper() for p in args.peers] if args.peers else None,
        db_path=args.db_path,
        download_dir=args.download_dir,
    )


if __name__ == "__main__":
    main()
