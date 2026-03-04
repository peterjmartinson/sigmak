#!/usr/bin/env python3
# DEPRECATED: use 'uv run sigmak peers --ticker TICKER --year YEAR' instead
"""Generate a Markdown peer comparison report for a target ticker and year.

This script now delegates to ``sigmak.reports.peer_report.run_peer_comparison``.

Preferred usage:
    uv run sigmak peers --ticker NVDA --year 2024

Legacy usage (still supported for backward compatibility):
    python scripts/generate_peer_comparison_report.py NVDA 2024
    python scripts/generate_peer_comparison_report.py NVDA 2024 --peers INTC AMD --max-peers 6
"""
from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a peer comparison report")
    parser.add_argument("ticker", help="Target ticker (e.g., NVDA)")
    parser.add_argument("year", type=int, help="Target filing year (e.g., 2024)")
    parser.add_argument("--peers", nargs="+", help="Explicit peer tickers (optional)")
    parser.add_argument("--max-peers", type=int, default=6, help="Max peers to include")
    parser.add_argument("--download-dir", type=str, default="./data/filings", help="Base download dir")
    parser.add_argument("--db-path", type=str, default="./database/sec_filings.db", help="Filings DB path")
    parser.add_argument("--output", type=str, default=None, help="Output markdown path (default: output/{TICKER}_Peer_Comparison_{YEAR}.md)")
    parser.add_argument("--db-only-classification", action="store_true", help="Use DB-only classification (no LLM calls)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    from sigmak.reports.peer_report import run_peer_comparison

    run_peer_comparison(
        ticker=args.ticker.upper(),
        year=args.year,
        max_peers=args.max_peers,
        explicit_peers=args.peers,
        db_only=bool(args.db_only_classification),
        db_path=args.db_path,
        download_dir=args.download_dir,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
