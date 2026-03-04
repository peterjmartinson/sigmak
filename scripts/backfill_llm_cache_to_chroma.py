#!/usr/bin/env python3
# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

"""
Backfill script to populate llm_risk_classification collection from output/*.json files.

This script now delegates to ``sigmak.cli.backfill.run``.

Usage:
    python scripts/backfill_llm_cache_to_chroma.py --dry-run
    python scripts/backfill_llm_cache_to_chroma.py --write
    python scripts/backfill_llm_cache_to_chroma.py --write --output-dir ./custom_output
"""

import sys
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Backfill LLM classifications from output/*.json to ChromaDB"
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run", action="store_true", help="Preview without writing"
    )
    mode_group.add_argument(
        "--write", action="store_true", help="Write to database"
    )
    parser.add_argument(
        "--output-dir", default="./output", help="Output JSON directory"
    )
    parser.add_argument(
        "--db-path", default="./database/sec_filings.db", help="SQLite path"
    )

    args = parser.parse_args()

    if not args.dry_run and not args.write:
        parser.error("Must specify either --dry-run or --write")

    from sigmak.cli.backfill import run

    run(
        write=args.write,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
        db_path=args.db_path,
    )


if __name__ == "__main__":
    main()
