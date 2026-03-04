"""Tests for the backfill CLI subcommand (issue #117).

Each test verifies exactly one behaviour (SRP).
"""

from unittest.mock import patch

from sigmak.__main__ import main


def test_backfill_main_dispatch_calls_cli_run() -> None:
    """main() dispatches to cli.backfill.run with dry_run=True."""
    with patch("sigmak.cli.backfill.run") as mock_run:
        main(["backfill", "--dry-run"])
        assert mock_run.call_args.kwargs["dry_run"] is True


def test_backfill_write_flag_forwarded() -> None:
    """--write is forwarded as write=True."""
    with patch("sigmak.cli.backfill.run") as mock_run:
        main(["backfill", "--write"])
        assert mock_run.call_args.kwargs["write"] is True


def test_backfill_output_dir_forwarded() -> None:
    """--output-dir value is forwarded as output_dir."""
    with patch("sigmak.cli.backfill.run") as mock_run:
        main(["backfill", "--dry-run", "--output-dir", "./custom"])
        assert mock_run.call_args.kwargs["output_dir"] == "./custom"


def test_backfill_no_ticker_required() -> None:
    """backfill subcommand must not raise SystemExit due to a missing --ticker."""
    with patch("sigmak.cli.backfill.run"):
        # Should not raise SystemExit(2) for missing required arg
        main(["backfill", "--dry-run"])
