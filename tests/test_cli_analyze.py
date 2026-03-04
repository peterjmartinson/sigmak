"""Tests for the `analyze` subcommand CLI wiring (Issue 116).

Each test verifies exactly one behaviour (SRP).
"""
from __future__ import annotations

import pytest
from unittest.mock import patch

from sigmak.__main__ import main


def test_analyze_main_dispatch_calls_cli_run() -> None:
    """main() dispatches to cli.analyze.run with ticker, year, html_path forwarded."""
    with patch("sigmak.cli.analyze.run") as mock_run:
        main(["analyze", "--ticker", "AAPL", "--year", "2024", "--html-path", "x.htm"])
        mock_run.assert_called_once()
        kw = mock_run.call_args.kwargs
        assert kw["ticker"] == "AAPL"
        assert kw["year"] == 2024
        assert kw["html_path"] == "x.htm"


def test_analyze_missing_html_exits_gracefully() -> None:
    """cli.analyze.run() exits with SystemExit(1) when html_path does not exist."""
    from sigmak.cli.analyze import run

    with pytest.raises(SystemExit) as exc_info:
        run(ticker="AAPL", year=2024, html_path="/nonexistent/path/filing.htm")
    assert exc_info.value.code == 1


def test_analyze_persist_path_forwarded() -> None:
    """--persist-path value is forwarded to cli.analyze.run as persist_path."""
    with patch("sigmak.cli.analyze.run") as mock_run:
        main(
            [
                "analyze",
                "--ticker", "AAPL",
                "--year", "2024",
                "--html-path", "x.htm",
                "--persist-path", "./mydb",
            ]
        )
        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["persist_path"] == "./mydb"
