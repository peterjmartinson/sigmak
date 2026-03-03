"""Tests for the `inspect` subcommand CLI wiring (Issue 105).

Each test verifies exactly one behaviour (SRP).
"""
from __future__ import annotations

from unittest.mock import patch

from sigmak.__main__ import main


def test_inspect_main_dispatch_calls_cli_run() -> None:
    """main() dispatches to cli.inspect_db.run with ticker forwarded."""
    with patch("sigmak.cli.inspect_db.run") as mock_run:
        main(["--ticker", "AAPL", "inspect"])
        mock_run.assert_called_once()
        kwargs = mock_run.call_args[1] if mock_run.call_args[1] else mock_run.call_args[0][0]
        # ticker is passed through **kwargs from argparse namespace
        call_kwargs = mock_run.call_args.kwargs if mock_run.call_args.kwargs else {}
        # argparse dispatches via run(**vars(args)), so positional or keyword
        all_args = {**dict(zip(["ticker"], mock_run.call_args.args)), **call_kwargs}
        assert all_args.get("ticker") == "AAPL"


def test_inspect_chroma_dir_forwarded() -> None:
    """--chroma-dir value is forwarded to cli.inspect_db.run as chroma_dir."""
    with patch("sigmak.cli.inspect_db.run") as mock_run:
        main(["--ticker", "AAPL", "inspect", "--chroma-dir", "./mydb"])
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args.kwargs if mock_run.call_args.kwargs else {}
        assert call_kwargs.get("chroma_dir") == "./mydb"
