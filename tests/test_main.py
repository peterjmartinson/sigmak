"""
Tests for src/sigmak/__main__.py — Issue 101: CLI Entry Point Scaffold.

Each test verifies exactly one behavior (SRP).
"""
import pytest

from sigmak.__main__ import build_parser, main


# ---------------------------------------------------------------------------
# build_parser() tests
# ---------------------------------------------------------------------------


def test_build_parser_requires_ticker() -> None:
    """Calling parse_args([]) without --ticker raises SystemExit."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_build_parser_accepts_ticker() -> None:
    """--ticker AAPL sets args.ticker == 'AAPL'."""
    parser = build_parser()
    args = parser.parse_args(["--ticker", "AAPL"])
    assert args.ticker == "AAPL"


def test_build_parser_use_llm_flag() -> None:
    """--use-llm sets args.use_llm == True."""
    parser = build_parser()
    args = parser.parse_args(["--ticker", "AAPL", "--use-llm"])
    assert args.use_llm is True


def test_build_parser_db_only_flag() -> None:
    """--db-only sets args.db_only == True."""
    parser = build_parser()
    args = parser.parse_args(["--ticker", "AAPL", "--db-only"])
    assert args.db_only is True


def test_build_parser_flags_are_mutually_exclusive() -> None:
    """Passing both --use-llm and --db-only raises SystemExit."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--ticker", "AAPL", "--use-llm", "--db-only"])


# ---------------------------------------------------------------------------
# Subcommand registration tests
# ---------------------------------------------------------------------------


def test_yoy_subcommand_registered() -> None:
    """parse_args(['--ticker', 'AAPL', 'yoy']) sets args.command == 'yoy'."""
    parser = build_parser()
    args = parser.parse_args(["--ticker", "AAPL", "yoy"])
    assert args.command == "yoy"


def test_peers_subcommand_registered() -> None:
    """parse_args(['--ticker', 'AAPL', 'peers', '--year', '2024']) sets args.command == 'peers'."""
    parser = build_parser()
    args = parser.parse_args(["--ticker", "AAPL", "peers", "--year", "2024"])
    assert args.command == "peers"


def test_download_subcommand_registered() -> None:
    """parse_args(['--ticker', 'AAPL', 'download']) sets args.command == 'download'."""
    parser = build_parser()
    args = parser.parse_args(["--ticker", "AAPL", "download"])
    assert args.command == "download"


def test_inspect_subcommand_registered() -> None:
    """parse_args(['--ticker', 'AAPL', 'inspect']) sets args.command == 'inspect'."""
    parser = build_parser()
    args = parser.parse_args(["--ticker", "AAPL", "inspect"])
    assert args.command == "inspect"


def test_render_subcommand_registered() -> None:
    """parse_args(['--ticker', 'AAPL', 'render', '--input', 'x.md']) sets args.command == 'render'."""
    parser = build_parser()
    args = parser.parse_args(["--ticker", "AAPL", "render", "--input", "x.md"])
    assert args.command == "render"


# ---------------------------------------------------------------------------
# main() tests
# ---------------------------------------------------------------------------


def test_no_subcommand_exits_zero() -> None:
    """main(['--ticker', 'AAPL']) with no subcommand prints help and exits 0."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--ticker", "AAPL"])
    assert exc_info.value.code == 0
