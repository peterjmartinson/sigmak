"""Tests for the `inspect` subcommand CLI wiring (Issue 105).

Each test verifies exactly one behaviour (SRP).
"""
from __future__ import annotations

from unittest.mock import patch

from sigmak.__main__ import main


def test_inspect_main_dispatch_calls_cli_run() -> None:
    """main() dispatches to cli.inspect_db.run; no --ticker required."""
    with patch("sigmak.cli.inspect_db.run") as mock_run:
        main(["inspect"])
        mock_run.assert_called_once()


def test_inspect_chroma_dir_forwarded() -> None:
    """--chroma-dir value is forwarded to cli.inspect_db.run as chroma_dir."""
    with patch("sigmak.cli.inspect_db.run") as mock_run:
        main(["inspect", "--chroma-dir", "./mydb"])
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args.kwargs if mock_run.call_args.kwargs else {}
        assert call_kwargs.get("chroma_dir") == "./mydb"
