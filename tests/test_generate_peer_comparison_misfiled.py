from pathlib import Path

from sigmak.reports.peer_report import locate_filing_html


def test_locate_misfiled_by_filename(tmp_path: Path):
    """If a 2024 filing is placed under the 2025 folder but the filename contains 2024,
    `locate_filing_html` should still find it.
    """
    ticker_dir = tmp_path / "SMCI"
    misfolder = ticker_dir / "2025"
    misfolder.mkdir(parents=True)

    # misfiled file: filename contains 2024 but it's under 2025
    misfile = misfolder / "smci-20241231.htm"
    misfile.write_text("<html><body>Filing date: 2024-12-31</body></html>", encoding="utf-8")

    # also create a proper 2025 file to ensure function chooses the 2024 one
    proper2025 = misfolder / "smci-20251231.htm"
    proper2025.write_text("<html><body>Filing date: 2025-12-31</body></html>", encoding="utf-8")

    found = locate_filing_html(ticker_dir, 2024)
    assert found is not None
    assert found.name == misfile.name
