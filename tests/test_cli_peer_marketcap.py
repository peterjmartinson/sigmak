"""Tests for the peer-marketcap CLI subcommand (issue #118).

Each test verifies exactly one behaviour (SRP).
"""

from unittest.mock import patch

from sigmak.__main__ import main


def test_peer_marketcap_dispatch_ticker() -> None:
    """main() forwards --ticker list as tickers=['AAPL', 'MSFT']."""
    with patch("sigmak.cli.peer_marketcap.run") as mock_run:
        main(["peer-marketcap", "--ticker", "AAPL", "MSFT"])
        assert mock_run.call_args.kwargs["tickers"] == ["AAPL", "MSFT"]


def test_peer_marketcap_dispatch_all() -> None:
    """--all is forwarded as all_peers=True."""
    with patch("sigmak.cli.peer_marketcap.run") as mock_run:
        main(["peer-marketcap", "--all"])
        assert mock_run.call_args.kwargs["all_peers"] is True


def test_peer_marketcap_delay_forwarded() -> None:
    """--delay value is forwarded as delay float."""
    with patch("sigmak.cli.peer_marketcap.run") as mock_run:
        main(["peer-marketcap", "--all", "--delay", "0.5"])
        assert mock_run.call_args.kwargs["delay"] == 0.5


def test_peer_marketcap_no_ticker_required() -> None:
    """peer-marketcap must not raise SystemExit due to a missing global --ticker."""
    with patch("sigmak.cli.peer_marketcap.run"):
        main(["peer-marketcap", "--all"])
