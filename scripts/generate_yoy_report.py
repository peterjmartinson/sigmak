#!/usr/bin/env python3
# DEPRECATED: use 'uv run sigmak yoy --ticker TICKER --years YEAR...' instead
"""Generate YoY risk analysis report for SEC 10-K filings.

This script now delegates to ``sigmak.reports.yoy_report.run_yoy_analysis``.

Preferred usage:
    uv run sigmak yoy --ticker HURC --years 2023 2024 2025

Legacy usage (still supported for backward compatibility):
    python scripts/generate_yoy_report.py HURC 2023 2024 2025
    python scripts/generate_yoy_report.py HURC 2023 2024 2025 --db-only-classification
"""
from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate YoY Risk Analysis Report for SEC 10-K filings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_yoy_report.py                          # Default: HURC 2023-2025
    python generate_yoy_report.py TSLA 2022 2023 2024     # Custom ticker and years
    python generate_yoy_report.py HURC 2023 2024 2025
        """,
    )
    parser.add_argument("ticker", nargs="?", default="HURC", help="Stock ticker (default: HURC)")
    parser.add_argument("years", nargs="*", type=int, help="Filing years (default: 2023 2024 2025)")
    parser.add_argument("--db-only-classification", action="store_true", help="Skip LLM; use DB-only classification")
    parser.add_argument("--db-only-similarity-threshold", type=float, default=0.8, help="Similarity threshold for DB-only matches (default: 0.8)")

    args = parser.parse_args()

    from sigmak.reports.yoy_report import run_yoy_analysis

    run_yoy_analysis(
        ticker=(args.ticker or "HURC").upper(),
        years=args.years if args.years else [2023, 2024, 2025],
        use_llm=False,
        db_only=args.db_only_classification,
        db_only_similarity_threshold=args.db_only_similarity_threshold,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
