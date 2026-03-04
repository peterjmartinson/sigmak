#!/usr/bin/env python3
# DEPRECATED: use 'uv run sigmak inspect' instead
"""Inspect persisted ChromaDB contents.

This script now delegates to ``sigmak.cli.inspect_db.run``.

Preferred usage:
    uv run sigmak inspect
    uv run sigmak inspect --chroma-dir ./database --max-sample 10

Legacy usage (still supported for backward compatibility):
    python scripts/inspect_chroma.py --dir chroma_db --max-sample 5
"""
from __future__ import annotations

import argparse
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect ChromaDB persistence directory")
    parser.add_argument("--dir", default="./database", help="Chroma persist directory (default: ./database)")
    parser.add_argument("--max-sample", type=int, default=5, help="Max sample rows per collection (default: 5)")
    args = parser.parse_args(argv)

    from sigmak.cli.inspect_db import run

    run(ticker="", chroma_dir=args.dir, max_sample=args.max_sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
