"""CLI handler for the ``backfill`` subcommand.

Delegates to :mod:`sigmak.backfill` (the library module that holds the
promoted helpers).

Usage::

    uv run sigmak backfill --dry-run
    uv run sigmak backfill --write
    uv run sigmak backfill --write --output-dir ./output --db-path ./database/sec_filings.db
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def run(
    write: bool = False,
    dry_run: bool = False,
    output_dir: str = "./output",
    db_path: str = "./database/sec_filings.db",
    **_: object,
) -> None:
    """Run the LLM classification backfill.

    Args:
        write:      Persist results to SQLite + ChromaDB.
        dry_run:    Preview changes without writing.
        output_dir: Directory containing ``results_*.json`` files.
        db_path:    Path to the SQLite database (overrides config.yaml).
        **_:        Absorbs extra kwargs injected by ``__main__``
                    (ticker, use_llm, db_only, …).
    """
    # Default to dry_run if neither flag given (safe default)
    if not write and not dry_run:
        logger.warning(
            "Neither --write nor --dry-run specified; defaulting to --dry-run."
        )
        dry_run = True

    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"Error: output directory not found: {output_path}", file=sys.stderr)
        sys.exit(1)

    from sigmak.config import get_settings
    from sigmak.backfill import run_backfill

    settings = get_settings()
    chroma_path = str(settings.chroma.persist_directory)
    embedding_model = settings.chroma.embedding_model

    # Resolve db_path: CLI override wins, else config default
    resolved_db_path = db_path if db_path != "./database/sec_filings.db" else str(
        settings.database.sqlite_path
    )

    mode_label = "DRY RUN" if dry_run else "WRITE"
    print(f"Backfill mode: {mode_label}")
    print(f"Output dir:    {output_dir}")
    print(f"Database:      {resolved_db_path}")
    print(f"ChromaDB:      {chroma_path}")
    print()

    stats = run_backfill(
        output_dir=output_dir,
        db_path=resolved_db_path,
        chroma_path=chroma_path,
        embedding_model=embedding_model,
        dry_run=dry_run,
    )

    print("\nBackfill Summary")
    print("=" * 40)
    print(stats)
    if dry_run:
        print("\nDRY RUN complete. Run with --write to persist changes.")
    else:
        print("\nBackfill complete.")
