"""Verify each kept script shim is syntactically valid after deprecation.

Each test verifies exactly one script is importable (SRP).
These tests protect against accidental syntax breakage during the
deprecation/slimming pass.
"""

import py_compile
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


def _check_syntax(name: str) -> None:
    """Compile-check a script by name, raising on any syntax error."""
    py_compile.compile(str(SCRIPTS_DIR / f"{name}.py"), doraise=True)


def test_generate_yoy_report_is_importable() -> None:
    _check_syntax("generate_yoy_report")


def test_generate_peer_comparison_report_is_importable() -> None:
    _check_syntax("generate_peer_comparison_report")


def test_download_peers_and_target_is_importable() -> None:
    _check_syntax("download_peers_and_target")


def test_inspect_chroma_is_importable() -> None:
    _check_syntax("inspect_chroma")


def test_md_to_pdf_is_importable() -> None:
    _check_syntax("md_to_pdf")


def test_backfill_llm_cache_to_chroma_is_importable() -> None:
    _check_syntax("backfill_llm_cache_to_chroma")


def test_populate_peer_marketcap_is_importable() -> None:
    _check_syntax("populate_peer_marketcap")
