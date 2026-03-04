#!/usr/bin/env python3
# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.
# DEPRECATED: use 'uv run sigmak render --input FILE' instead

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


# HTML_TEMPLATE and convert_md_to_pdf now live in sigmak.reports.pdf_renderer.
# This re-export preserves backward compatibility for any callers.
def convert_md_to_pdf(
    md_path: Path,
    out_pdf: Path,
    css_path: Path,
    title: str | None = None,
) -> None:
    """Thin wrapper — delegates to ``sigmak.reports.pdf_renderer.convert_md_to_pdf``."""
    from sigmak.reports.pdf_renderer import convert_md_to_pdf as _impl

    _impl(md_path, out_pdf, css_path, title)


def main() -> None:
    """Delegate to ``sigmak.cli.render.run``."""
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

    from sigmak.cli.render import run

    run(
        input_path=str(args.md),
        output_path=str(args.out) if args.out else None,
        css_path=str(args.css),
        title=args.title,
    )


if __name__ == "__main__":
    main()
