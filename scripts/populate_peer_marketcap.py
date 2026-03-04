#!/usr/bin/env python3
# DEPRECATED: use 'uv run sigmak peer-marketcap --ticker TICKER...' or '--all' instead
"""Populate market_cap for peers in `database/sec_filings.db`.

This script now delegates to ``sigmak.cli.peer_marketcap.run``.

Usage:
  python scripts/populate_peer_marketcap.py --tickers AAPL,MSFT
  python scripts/populate_peer_marketcap.py --all
"""
import argparse
from dotenv import load_dotenv

load_dotenv()


def main(argv=None):
    p = argparse.ArgumentParser(description="Populate market_cap for peers using yfinance")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--tickers", help="Comma-separated tickers to update")
    group.add_argument("--all", action="store_true", help="Update all peers in DB")
    p.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between requests")
    p.add_argument("--db-path", default="./database/sec_filings.db", help="Path to filings DB")

    args = p.parse_args(argv)

    # Convert comma-separated --tickers to a list for the CLI handler
    tickers = None
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    from sigmak.cli.peer_marketcap import run

    run(
        tickers=tickers,
        all_peers=args.all,
        delay=args.delay,
        db_path=args.db_path,
    )


if __name__ == "__main__":
    main()
