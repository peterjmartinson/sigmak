"""
Tests for src/sigmak/__main__.py — Issue 101 / Issue 114.

Each test verifies exactly one behavior (SRP).
"""
import pytest

from sigmak.__main__ import build_parser, main


# ---------------------------------------------------------------------------
# build_parser() tests
# ---------------------------------------------------------------------------


def test_build_parser_requires_ticker() -> None:
    """Calling 'yoy' without --ticker raises SystemExit."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["yoy"])


def test_build_parser_accepts_ticker() -> None:
    """--ticker AAPL under yoy sets args.ticker == 'AAPL'."""
    parser = build_parser()
    args = parser.parse_args(["yoy", "--ticker", "AAPL"])
    assert args.ticker == "AAPL"


def test_build_parser_use_llm_flag() -> None:
    """--use-llm under yoy sets args.use_llm == True."""
    parser = build_parser()
    args = parser.parse_args(["yoy", "--ticker", "AAPL", "--use-llm"])
    assert args.use_llm is True


def test_build_parser_db_only_flag() -> None:
    """--db-only under yoy sets args.db_only == True."""
    parser = build_parser()
    args = parser.parse_args(["yoy", "--ticker", "AAPL", "--db-only"])
    assert args.db_only is True


def test_build_parser_flags_are_mutually_exclusive() -> None:
    """Passing both --use-llm and --db-only under yoy raises SystemExit."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["yoy", "--ticker", "AAPL", "--use-llm", "--db-only"])


# ---------------------------------------------------------------------------
# Subcommand registration tests
# ---------------------------------------------------------------------------


def test_yoy_subcommand_registered() -> None:
    """parse_args(['yoy', '--ticker', 'AAPL']) sets args.command == 'yoy'."""
    parser = build_parser()
    args = parser.parse_args(["yoy", "--ticker", "AAPL"])
    assert args.command == "yoy"


def test_peers_subcommand_registered() -> None:
    """parse_args(['peers', '--ticker', 'AAPL', '--year', '2024']) sets args.command == 'peers'."""
    parser = build_parser()
    args = parser.parse_args(["peers", "--ticker", "AAPL", "--year", "2024"])
    assert args.command == "peers"


def test_download_subcommand_registered() -> None:
    """parse_args(['download', '--ticker', 'AAPL']) sets args.command == 'download'."""
    parser = build_parser()
    args = parser.parse_args(["download", "--ticker", "AAPL"])
    assert args.command == "download"


def test_inspect_subcommand_registered() -> None:
    """parse_args(['inspect']) sets args.command == 'inspect' without requiring --ticker."""
    parser = build_parser()
    args = parser.parse_args(["inspect"])
    assert args.command == "inspect"


def test_inspect_runs_without_ticker() -> None:
    """inspect subcommand does not require --ticker."""
    parser = build_parser()
    # Must not raise
    args = parser.parse_args(["inspect"])
    assert not hasattr(args, "ticker") or args.ticker is None


def test_render_subcommand_registered() -> None:
    """parse_args(['render', '--input', 'x.md']) sets args.command == 'render'."""
    parser = build_parser()
    args = parser.parse_args(["render", "--input", "x.md"])
    assert args.command == "render"


def test_render_runs_without_ticker() -> None:
    """render subcommand does not require --ticker."""
    parser = build_parser()
    # Must not raise
    args = parser.parse_args(["render", "--input", "x.md"])
    assert not hasattr(args, "ticker") or args.ticker is None


# ---------------------------------------------------------------------------
# main() tests
# ---------------------------------------------------------------------------


def test_no_subcommand_exits_zero() -> None:
    """main([]) with no subcommand prints help and exits 0."""
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 0
