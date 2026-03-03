"""
Tests for Issue 102: yoy CLI subcommand wiring.

Each test verifies exactly one behavior (SRP).
"""
from unittest.mock import patch, MagicMock

from sigmak.__main__ import main


def test_yoy_run_calls_run_yoy_analysis() -> None:
    """cli.yoy.run() delegates to sigmak.reports.yoy_report.run_yoy_analysis."""
    with patch("sigmak.reports.yoy_report.run_yoy_analysis") as mock_fn:
        from sigmak.cli.yoy import run
        run(ticker="AAPL", years=[2023, 2024], use_llm=False, db_only=False)
        mock_fn.assert_called_once_with(
            ticker="AAPL",
            years=[2023, 2024],
            use_llm=False,
            db_only=False,
        )


def test_yoy_main_dispatch_calls_cli_run() -> None:
    """main() dispatches to cli.yoy.run with ticker forwarded."""
    with patch("sigmak.cli.yoy.run") as mock_run:
        main(["--ticker", "AAPL", "yoy", "--years", "2023", "2024"])
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["ticker"] == "AAPL"


def test_yoy_default_years() -> None:
    """main() passes years=[2023, 2024, 2025] when --years is not supplied."""
    with patch("sigmak.cli.yoy.run") as mock_run:
        main(["--ticker", "AAPL", "yoy"])
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["years"] == [2023, 2024, 2025]


def test_yoy_db_only_flag_passed() -> None:
    """--db-only global flag is forwarded as db_only=True to cli.yoy.run."""
    with patch("sigmak.cli.yoy.run") as mock_run:
        main(["--ticker", "AAPL", "--db-only", "yoy"])
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["db_only"] is True
