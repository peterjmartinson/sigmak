"""Tests for the `render` subcommand CLI wiring (Issue 105).

Each test verifies exactly one behaviour (SRP).
"""
from __future__ import annotations

import pytest
from unittest.mock import patch

from sigmak.__main__ import main


def test_render_main_dispatch_calls_cli_run() -> None:
    """main() dispatches to cli.render.run with input_path forwarded; no --ticker needed."""
    with patch("sigmak.cli.render.run") as mock_run:
        main(["render", "--input", "output/test.md"])
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args.kwargs if mock_run.call_args.kwargs else {}
        assert call_kwargs.get("input_path") == "output/test.md"


def test_render_missing_weasyprint_exits_gracefully(tmp_path: "Path") -> None:  # type: ignore[name-defined]
    """cli.render.run exits with SystemExit(1) when weasyprint is unavailable."""
    from pathlib import Path

    md = tmp_path / "x.md"
    md.write_text("# test")
    css = tmp_path / "report.css"
    css.write_text("body {}")

    with patch(
        "sigmak.reports.pdf_renderer.convert_md_to_pdf",
        side_effect=ImportError("No module named 'weasyprint'"),
    ):
        from sigmak.cli.render import run

        with pytest.raises(SystemExit) as exc_info:
            run(input_path=str(md), css_path=str(css))
        assert exc_info.value.code == 1


def test_render_output_path_forwarded() -> None:
    """--output value is forwarded to cli.render.run as output_path."""
    with patch("sigmak.cli.render.run") as mock_run:
        main(
            [
                "render",
                "--input",
                "output/test.md",
                "--output",
                "output/test.pdf",
            ]
        )
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args.kwargs if mock_run.call_args.kwargs else {}
        assert call_kwargs.get("output_path") == "output/test.pdf"
