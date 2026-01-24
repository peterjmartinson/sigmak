#!/usr/bin/env python3
# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

"""
Simple, easy-to-read Markdown → PDF converter using WeasyPrint.

This script converts Markdown reports (from generate_yoy_report.py) into
styled PDFs suitable for distribution. The implementation is deliberately
simple and explicit so you can modify styling and layout later without
confusion.

Usage:
    python scripts/md_to_pdf.py output/TSLA_YoY_Risk_Analysis_2023_2025.md
    python scripts/md_to_pdf.py output/report.md -o output/report.pdf
    python scripts/md_to_pdf.py output/report.md --title "Custom Title"

Dependencies:
    pip install weasyprint markdown jinja2

    # System deps (Debian/Ubuntu):
    sudo apt install build-essential libffi-dev libcairo2 libpango-1.0-0 libgdk-pixbuf2.0-0
"""

from pathlib import Path
import argparse
import sys
from markdown import markdown
from jinja2 import Template
from weasyprint import HTML, CSS


# Simple HTML template wrapper for the Markdown content
HTML_TEMPLATE = """<!doctype html>
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
    title: str | None = None
) -> None:
    """
    Convert Markdown file to styled PDF.
    
    Args:
        md_path: Input Markdown file
        out_pdf: Output PDF file
        css_path: CSS stylesheet for PDF styling
        title: Document title (defaults to filename stem)
    
    The conversion process:
        1. Read Markdown text
        2. Convert to HTML using Python-Markdown
        3. Wrap in minimal HTML template
        4. Render to PDF with CSS styling via WeasyPrint
    """
    # Read Markdown source
    md_text = md_path.read_text(encoding="utf-8")
    
    # Convert Markdown → HTML
    # Extensions: fenced_code (```) | tables | toc (auto table of contents)
    html_body = markdown(
        md_text,
        extensions=["fenced_code", "tables", "toc"]
    )
    
    # Wrap in minimal HTML template
    template = Template(HTML_TEMPLATE)
    html = template.render(
        title=title or md_path.stem,
        content=html_body
    )
    
    # Render HTML → PDF with styling
    # base_url allows relative image paths in Markdown to resolve correctly
    HTML(string=html, base_url=str(md_path.parent)).write_pdf(
        str(out_pdf),
        stylesheets=[CSS(filename=str(css_path))]
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert Markdown report to styled PDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/md_to_pdf.py output/TSLA_YoY_Risk_Analysis_2023_2025.md
  python scripts/md_to_pdf.py output/report.md -o output/custom_name.pdf
  python scripts/md_to_pdf.py output/report.md --title "Q4 2024 Risk Analysis"
        """
    )
    
    parser.add_argument(
        "md",
        type=Path,
        help="Input Markdown file to convert"
    )
    parser.add_argument(
        "-o", "--out",
        type=Path,
        default=None,
        help="Output PDF path (default: same as input with .pdf extension)"
    )
    parser.add_argument(
        "--css",
        type=Path,
        default=Path("styles/report.css"),
        help="CSS stylesheet for PDF styling (default: styles/report.css)"
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Document title (default: input filename)"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not args.md.exists():
        print(f"Error: Input file not found: {args.md}", file=sys.stderr)
        sys.exit(1)
    
    # Validate CSS file exists
    if not args.css.exists():
        print(f"Error: CSS file not found: {args.css}", file=sys.stderr)
        print(f"Hint: Create a default CSS at styles/report.css", file=sys.stderr)
        sys.exit(1)
    
    # Default output path
    out_pdf = args.out or args.md.with_suffix(".pdf")
    
    # Convert
    try:
        convert_md_to_pdf(args.md, out_pdf, args.css, args.title)
        print(f"✓ Created PDF: {out_pdf}")
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
