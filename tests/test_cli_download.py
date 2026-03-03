"""
Tests for Issue 104: download CLI subcommand wiring.

Each test verifies exactly one behavior (SRP).
"""
from unittest.mock import patch

from sigmak.__main__ import main


def test_download_run_signature_accepts_required_args() -> None:
    """run() can be called with required args without raising."""
    from sigmak.cli.download import run

    with (
        patch("sigmak.cli.download.TenKDownloader"),
        patch("sigmak.cli.download.PeerDiscoveryService"),
        patch("sigmak.cli.download._download_one", return_value=("NVDA", "downloaded")),
    ):
        run(ticker="NVDA", years=[2024], include_peers=False, db_only=False)


def test_download_main_dispatch_calls_cli_run() -> None:
    """main() dispatches to cli.download.run with ticker forwarded."""
    with patch("sigmak.cli.download.run") as mock_run:
        main(["--ticker", "NVDA", "download"])
        mock_run.assert_called_once()
        assert mock_run.call_args[1]["ticker"] == "NVDA"


def test_download_include_peers_flag() -> None:
    """--include-peers is forwarded as include_peers=True to cli.download.run."""
    with patch("sigmak.cli.download.run") as mock_run:
        main(["--ticker", "NVDA", "download", "--include-peers"])
        mock_run.assert_called_once()
        assert mock_run.call_args[1]["include_peers"] is True


def test_download_years_forwarded() -> None:
    """--years 2023 2024 is forwarded as years=[2023, 2024] to cli.download.run."""
    with patch("sigmak.cli.download.run") as mock_run:
        main(["--ticker", "NVDA", "download", "--years", "2023", "2024"])
        mock_run.assert_called_once()
        assert mock_run.call_args[1]["years"] == [2023, 2024]
