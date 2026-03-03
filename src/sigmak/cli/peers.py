"""CLI handler for the `peers` subcommand."""
from __future__ import annotations

from typing import List


def run(
    ticker: str,
    year: int,
    max_peers: int,
    explicit_peers: List[str] | None,
    db_only: bool,
    use_sic_only: bool = False,
    **_: object,
) -> None:
    """Delegate to ``sigmak.reports.peer_report.run_peer_comparison``."""
    from sigmak.reports.peer_report import run_peer_comparison

    run_peer_comparison(
        ticker=ticker,
        year=year,
        max_peers=max_peers,
        explicit_peers=explicit_peers,
        db_only=db_only,
        use_sic_only=use_sic_only,
    )
