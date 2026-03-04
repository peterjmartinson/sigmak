#!/usr/bin/env python3
"""
CLI script to analyze SEC filings.

Usage:
    python analyze_filing.py data/filings/tsla_2024_10k.html TSLA 2024

This script now delegates to ``sigmak.cli.analyze.run``.
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    if len(sys.argv) != 4:
        print("Usage: python analyze_filing.py <html_path> <ticker> <year>")
        print("Example: python analyze_filing.py data/sample_10k.html AAPL 2025")
        sys.exit(1)

    html_path = sys.argv[1]
    ticker = sys.argv[2].upper()
    year = int(sys.argv[3])

    from sigmak.cli.analyze import run

    run(
        ticker=ticker,
        year=year,
        html_path=html_path,
        use_llm=bool(os.getenv("GOOGLE_API_KEY")),
    )


if __name__ == "__main__":
    main()
