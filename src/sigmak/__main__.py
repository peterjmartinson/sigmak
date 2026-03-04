"""
Entry point for `uv run sigmak` / `python -m sigmak`.

Usage
-----
    uv run sigmak yoy --ticker AAPL
    uv run sigmak peers --ticker AAPL --year 2024
    uv run sigmak download --ticker AAPL
    uv run sigmak inspect
    uv run sigmak render --input output/AAPL_YoY.md
"""
from __future__ import annotations

import argparse
import os
import sys


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser.

    This is a standalone function so it can be imported and tested without
    running main().
    """
    parser = argparse.ArgumentParser(
        prog="sigmak",
        description="Proprietary Risk Scoring CLI for SEC 10-K/Q filings.",
    )

    subparsers = parser.add_subparsers(dest="command")

    yoy_parser = subparsers.add_parser("yoy", help="Year-over-year risk analysis.")
    yoy_parser.add_argument(
        "--ticker",
        required=True,
        metavar="TICKER",
        help="Target company ticker symbol (required).",
    )
    yoy_mode_group = yoy_parser.add_mutually_exclusive_group()
    yoy_mode_group.add_argument(
        "--use-llm",
        action="store_true",
        default=False,
        help="Use LLM for classification (requires GOOGLE_API_KEY).",
    )
    yoy_mode_group.add_argument(
        "--db-only",
        action="store_true",
        default=False,
        help="Use ChromaDB only; no LLM calls.",
    )
    yoy_parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2023, 2024, 2025],
        metavar="YEAR",
        help="Filing years to analyse (default: 2023 2024 2025).",
    )
    peers_parser = subparsers.add_parser("peers", help="Peer comparison report.")
    peers_parser.add_argument(
        "--ticker",
        required=True,
        metavar="TICKER",
        help="Target company ticker symbol (required).",
    )
    peers_mode_group = peers_parser.add_mutually_exclusive_group()
    peers_mode_group.add_argument(
        "--use-llm",
        action="store_true",
        default=False,
        help="Use LLM for classification (requires GOOGLE_API_KEY).",
    )
    peers_mode_group.add_argument(
        "--db-only",
        action="store_true",
        default=False,
        help="Use ChromaDB only; no LLM calls.",
    )
    peers_parser.add_argument(
        "--year",
        type=int,
        required=True,
        metavar="YEAR",
        help="Filing year to analyse.",
    )
    peers_parser.add_argument(
        "--max-peers",
        type=int,
        default=6,
        dest="max_peers",
        metavar="N",
        help="Maximum number of peers to include (default: 6).",
    )
    peers_parser.add_argument(
        "--peers",
        nargs="*",
        dest="explicit_peers",
        default=None,
        metavar="TICKER",
        help="Explicit peer tickers (overrides auto-discovery).",
    )
    peers_parser.add_argument(
        "--sic-only",
        action="store_true",
        dest="use_sic_only",
        default=False,
        help="Use SIC/EDGAR peer selection instead of yfinance (default).",
    )
    download_parser = subparsers.add_parser("download", help="Download SEC filings.")
    download_parser.add_argument(
        "--ticker",
        required=True,
        metavar="TICKER",
        help="Target company ticker symbol (required).",
    )
    download_parser.add_argument(
        "--db-only",
        action="store_true",
        default=False,
        dest="db_only",
        help="Use ChromaDB only; no LLM calls.",
    )
    download_parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        dest="years",
        metavar="YEAR",
        help="Filing years to download (default: latest available for each ticker).",
    )
    download_parser.add_argument(
        "--include-peers",
        action="store_true",
        default=False,
        dest="include_peers",
        help="Also download filings for auto-discovered peers (uses yfinance).",
    )
    download_parser.add_argument(
        "--max-peers",
        type=int,
        default=6,
        dest="max_peers",
        metavar="N",
        help="Maximum number of peers to download when --include-peers is set (default: 6).",
    )
    inspect_parser = subparsers.add_parser("inspect", help="Inspect the local database.")
    inspect_parser.add_argument(
        "--chroma-dir",
        default="./database",
        dest="chroma_dir",
        metavar="PATH",
        help="ChromaDB persistence directory (default: ./database).",
    )
    inspect_parser.add_argument(
        "--max-sample",
        type=int,
        default=5,
        dest="max_sample",
        metavar="N",
        help="Maximum sample rows per collection (default: 5).",
    )

    render_parser = subparsers.add_parser("render", help="Render a Markdown report to PDF.")
    render_parser.add_argument(
        "--input",
        required=True,
        dest="input_path",
        metavar="PATH",
        help="Input Markdown file to convert.",
    )
    render_parser.add_argument(
        "--output",
        default=None,
        dest="output_path",
        metavar="PATH",
        help="Output PDF path (default: same stem as input).",
    )
    render_parser.add_argument(
        "--css",
        default="styles/report.css",
        dest="css_path",
        metavar="PATH",
        help="CSS stylesheet for PDF styling (default: styles/report.css).",
    )
    render_parser.add_argument(
        "--title",
        default=None,
        help="Document title (default: input filename stem).",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch to the appropriate CLI module.

    Parameters
    ----------
    argv:
        Argument list to parse. Defaults to sys.argv[1:] when None.
    """
    os.environ.setdefault("SIGMAK_PEER_YFINANCE_ENABLED", "true")

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Provide safe defaults for flags not present in utility subparsers
    # (inspect, render do not declare ticker/use_llm/db_only)
    kwargs = vars(args)
    kwargs.setdefault("ticker", None)
    kwargs.setdefault("use_llm", False)
    kwargs.setdefault("db_only", False)

    if args.command == "yoy":
        from sigmak.cli.yoy import run
        run(**kwargs)
    elif args.command == "peers":
        from sigmak.cli.peers import run
        run(**kwargs)
    elif args.command == "download":
        from sigmak.cli.download import run
        run(**kwargs)
    elif args.command == "inspect":
        from sigmak.cli.inspect_db import run
        run(**kwargs)
    elif args.command == "render":
        from sigmak.cli.render import run
        run(**kwargs)


if __name__ == "__main__":
    main()
