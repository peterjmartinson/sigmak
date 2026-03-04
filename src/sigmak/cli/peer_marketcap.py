"""CLI handler for the ``peer-marketcap`` subcommand.

Delegates to :func:`sigmak.filings_db.populate_market_cap`.

Usage::

    uv run sigmak peer-marketcap --ticker AAPL MSFT
    uv run sigmak peer-marketcap --all
    uv run sigmak peer-marketcap --all --delay 0.5 --db-path ./database/sec_filings.db
"""

import sys
import logging

logger = logging.getLogger(__name__)


def run(
    tickers: list[str] | None = None,
    all_peers: bool = False,
    delay: float = 1.0,
    db_path: str = "./database/sec_filings.db",
    **_: object,
) -> None:
    """Populate market-cap data for peers in the filings database.

    Args:
        tickers:   Explicit list of tickers to update. Mutually exclusive
                   with ``all_peers``.
        all_peers: When ``True``, update every peer present in the DB.
        delay:     Seconds to wait between yfinance requests (rate-limit).
        db_path:   Path to the SQLite filings database.
        **_:       Absorbs extra kwargs injected by ``__main__``
                   (ticker, use_llm, db_only, …).
    """
    resolved_tickers: list[str] | None = tickers if not all_peers else None

    try:
        from sigmak.filings_db import populate_market_cap
    except ImportError as exc:
        print(f"Error: could not import filings_db: {exc}", file=sys.stderr)
        sys.exit(1)

    label = "all peers" if all_peers else f"{resolved_tickers}"
    print(f"Fetching market-cap for {label} (delay={delay}s, db={db_path})")

    try:
        updated = populate_market_cap(db_path, tickers=resolved_tickers, delay=delay)
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    print(f"Updated market_cap for {updated} peer(s).")
