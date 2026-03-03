"""
Tests for Issue 103: peers CLI subcommand wiring.

Each test verifies exactly one behavior (SRP).
"""
from unittest.mock import patch

from sigmak.__main__ import main


def test_peers_run_calls_run_peer_comparison() -> None:
    """cli.peers.run() delegates to sigmak.reports.peer_report.run_peer_comparison."""
    with patch("sigmak.reports.peer_report.run_peer_comparison") as mock_fn:
        from sigmak.cli.peers import run

        run(ticker="NVDA", year=2024, max_peers=6, explicit_peers=None, db_only=False)
        mock_fn.assert_called_once_with(
            ticker="NVDA",
            year=2024,
            max_peers=6,
            explicit_peers=None,
            db_only=False,
        )


def test_peers_main_dispatch_calls_cli_run() -> None:
    """main() dispatches to cli.peers.run with ticker and year forwarded."""
    with patch("sigmak.cli.peers.run") as mock_run:
        main(["--ticker", "NVDA", "peers", "--year", "2024"])
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["ticker"] == "NVDA"
        assert call_kwargs["year"] == 2024


def test_peers_explicit_peers_forwarded() -> None:
    """--peers AAPL MSFT is forwarded as explicit_peers=['AAPL', 'MSFT']."""
    with patch("sigmak.cli.peers.run") as mock_run:
        main(["--ticker", "NVDA", "peers", "--year", "2024", "--peers", "AAPL", "MSFT"])
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["explicit_peers"] == ["AAPL", "MSFT"]


def test_peers_db_only_flag_forwarded() -> None:
    """--db-only global flag is forwarded as db_only=True to cli.peers.run."""
    with patch("sigmak.cli.peers.run") as mock_run:
        main(["--ticker", "NVDA", "--db-only", "peers", "--year", "2024"])
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["db_only"] is True
