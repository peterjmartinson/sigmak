"""CLI handler for the `render` subcommand.

Converts a Markdown report to a styled PDF via WeasyPrint.
"""
from __future__ import annotations

import sys
from pathlib import Path


def run(
    input_path: str,
    output_path: str | None = None,
    css_path: str = "styles/report.css",
    title: str | None = None,
    **_: object,
) -> None:
    """Render *input_path* (Markdown) to a PDF.

    Parameters
    ----------
    input_path:
        Path to the input ``.md`` file.
    output_path:
        Destination ``.pdf`` path.  Defaults to *input_path* with ``.pdf``
        extension.
    css_path:
        Path to the CSS stylesheet (default: ``styles/report.css``).
    title:
        Document title embedded in the PDF.  Defaults to the input filename
        stem.
    """
    md = Path(input_path)
    if not md.exists():
        print(f"Error: Input file not found: {md}", file=sys.stderr)
        sys.exit(1)

    css = Path(css_path)
    if not css.exists():
        print(f"Error: CSS file not found: {css}", file=sys.stderr)
        print("Hint: create a stylesheet at styles/report.css", file=sys.stderr)
        sys.exit(1)

    out_pdf = Path(output_path) if output_path else md.with_suffix(".pdf")

    from sigmak.reports.pdf_renderer import convert_md_to_pdf

    try:
        convert_md_to_pdf(md, out_pdf, css, title)
    except ImportError:
        print(
            "Error: PDF rendering requires the 'pdf' optional dependencies.\n"
            "Install them with:  uv pip install sigmak[pdf]",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as exc:
        print(f"Error during PDF conversion: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Created PDF: {out_pdf}")
