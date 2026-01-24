# PDF Styling

This directory contains CSS stylesheets for converting Markdown reports to PDFs.

## Files

- `report.css` — Default stylesheet for SEC risk analysis reports
  - Professional serif typography (Georgia)
  - NASA blue (#0b3d91) headers for authority
  - Clean code blocks and tables
  - Letter-size pages with 1" margins

## Customizing Styles

Edit `report.css` to adjust:
- **Typography**: Change `font-family`, `font-size`, `line-height`
- **Colors**: Update `color` values for headers, body, code backgrounds
- **Layout**: Modify `@page` margins, page breaks, spacing
- **Tables**: Adjust borders, alternating row colors, padding

## Adding New Stylesheets

Create a new CSS file (e.g., `quarterly_report.css`) and pass it to the converter:

```bash
python scripts/md_to_pdf.py output/report.md --css styles/quarterly_report.css
```

## CSS Reference

WeasyPrint supports most CSS 2.1 and CSS 3 properties, plus special `@page` rules for PDF layout:
- Documentation: https://doc.courtbouillon.org/weasyprint/stable/
- Paged Media: https://www.w3.org/TR/css-page-3/
