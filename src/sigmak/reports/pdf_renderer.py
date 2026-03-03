"""PDF rendering utilities for SigmaK reports.

``weasyprint``, ``markdown``, and ``jinja2`` are optional dependencies
(extras group ``pdf``).  All three are imported lazily inside
``convert_md_to_pdf`` so this module can be imported without them installed.
"""
from __future__ import annotations

from pathlib import Path

_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>{{ title }}</title>
</head>
<body>
{{ content }}
</body>
</html>
"""


def convert_md_to_pdf(
    md_path: Path,
    out_pdf: Path,
    css_path: Path,
    title: str | None = None,
) -> None:
    """Convert a Markdown file to a styled PDF.

    Parameters
    ----------
    md_path:
        Input Markdown file.
    out_pdf:
        Output PDF file.
    css_path:
        CSS stylesheet for PDF styling.
    title:
        Document title; defaults to the input filename stem.

    Raises
    ------
    ImportError
        If ``weasyprint``, ``markdown``, or ``jinja2`` are not installed.
    """
    from markdown import markdown  # type: ignore[import-untyped]
    from jinja2 import Template  # type: ignore[import-untyped]
    from weasyprint import HTML, CSS  # type: ignore[import-untyped]

    md_text = md_path.read_text(encoding="utf-8")
    html_body = markdown(md_text, extensions=["fenced_code", "tables", "toc"])
    template = Template(_HTML_TEMPLATE)
    html = template.render(title=title or md_path.stem, content=html_body)
    HTML(string=html, base_url=str(md_path.parent)).write_pdf(
        str(out_pdf),
        stylesheets=[CSS(filename=str(css_path))],
    )
