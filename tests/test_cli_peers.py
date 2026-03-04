"""
Tests for Issue 103 / Issue 110: peers CLI subcommand wiring and yfinance peer selection.

Each test verifies exactly one behavior (SRP).
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sigmak.__main__ import main


def test_peers_run_calls_run_peer_comparison() -> None:
    """cli.peers.run() delegates to sigmak.reports.peer_report.run_peer_comparison."""
    with patch("sigmak.reports.peer_report.run_peer_comparison") as mock_fn:
        from sigmak.cli.peers import run

        run(ticker="NVDA", year=2024, max_peers=6, explicit_peers=None, db_only=False, use_sic_only=False)
        mock_fn.assert_called_once_with(
            ticker="NVDA",
            year=2024,
            max_peers=6,
            explicit_peers=None,
            db_only=False,
            use_sic_only=False,
        )


def test_peers_main_dispatch_calls_cli_run() -> None:
    """main() dispatches to cli.peers.run with ticker and year forwarded."""
    with patch("sigmak.cli.peers.run") as mock_run:
        main(["peers", "--ticker", "NVDA", "--year", "2024"])
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["ticker"] == "NVDA"
        assert call_kwargs["year"] == 2024


def test_peers_explicit_peers_forwarded() -> None:
    """--peers AAPL MSFT is forwarded as explicit_peers=['AAPL', 'MSFT']."""
    with patch("sigmak.cli.peers.run") as mock_run:
        main(["peers", "--ticker", "NVDA", "--year", "2024", "--peers", "AAPL", "MSFT"])
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["explicit_peers"] == ["AAPL", "MSFT"]


def test_peers_db_only_flag_forwarded() -> None:
    """--db-only flag is forwarded as db_only=True to cli.peers.run."""
    with patch("sigmak.cli.peers.run") as mock_run:
        main(["peers", "--ticker", "NVDA", "--db-only", "--year", "2024"])
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["db_only"] is True


def test_peers_sic_only_flag_registered() -> None:
    """--sic-only is forwarded as use_sic_only=True; defaults to False when absent."""
    with patch("sigmak.cli.peers.run") as mock_run:
        main(["peers", "--ticker", "AAPL", "--year", "2024", "--sic-only"])
        mock_run.assert_called_once()
        assert mock_run.call_args[1]["use_sic_only"] is True

    with patch("sigmak.cli.peers.run") as mock_run:
        main(["peers", "--ticker", "AAPL", "--year", "2024"])
        mock_run.assert_called_once()
        assert mock_run.call_args[1]["use_sic_only"] is False


def test_run_peer_comparison_defaults_to_yfinance() -> None:
    """run_peer_comparison calls get_peers_via_yfinance when use_sic_only=False."""
    from sigmak.reports.peer_report import run_peer_comparison

    mock_result = MagicMock()
    mock_result.ticker = "NVDA"
    mock_result.filing_year = 2024
    mock_result.risks = []
    mock_result.metadata = {}

    with (
        patch("sigmak.reports.peer_report.TenKDownloader"),
        patch("sigmak.reports.peer_report.IntegrationPipeline"),
        patch("sigmak.reports.peer_report.PeerDiscoveryService") as mock_svc_cls,
        patch("sigmak.reports.peer_report.ensure_filing", return_value=MagicMock()),
        patch("sigmak.reports.peer_report.load_or_analyze_with_cache", return_value=mock_result),
        patch("sigmak.reports.peer_report.generate_markdown_report"),
    ):
        mock_svc = mock_svc_cls.return_value
        mock_svc.get_peers_via_yfinance.return_value = []

        run_peer_comparison(
            ticker="NVDA", year=2024, max_peers=6, explicit_peers=None,
            db_only=False, use_sic_only=False,
        )

        mock_svc.get_peers_via_yfinance.assert_called_once()
        mock_svc.find_peers_for_ticker.assert_not_called()


def test_run_peer_comparison_sic_only_calls_find_peers_for_ticker() -> None:
    """run_peer_comparison calls find_peers_for_ticker when use_sic_only=True."""
    from sigmak.reports.peer_report import run_peer_comparison

    mock_result = MagicMock()
    mock_result.ticker = "NVDA"
    mock_result.filing_year = 2024
    mock_result.risks = []
    mock_result.metadata = {}

    with (
        patch("sigmak.reports.peer_report.TenKDownloader"),
        patch("sigmak.reports.peer_report.IntegrationPipeline"),
        patch("sigmak.reports.peer_report.PeerDiscoveryService") as mock_svc_cls,
        patch("sigmak.reports.peer_report.ensure_filing", return_value=MagicMock()),
        patch("sigmak.reports.peer_report.load_or_analyze_with_cache", return_value=mock_result),
        patch("sigmak.reports.peer_report.generate_markdown_report"),
    ):
        mock_svc = mock_svc_cls.return_value
        mock_svc.find_peers_for_ticker.return_value = []

        run_peer_comparison(
            ticker="NVDA", year=2024, max_peers=6, explicit_peers=None,
            db_only=False, use_sic_only=True,
        )

        mock_svc.find_peers_for_ticker.assert_called_once()
        mock_svc.get_peers_via_yfinance.assert_not_called()
