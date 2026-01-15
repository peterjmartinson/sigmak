import sqlite3
from pathlib import Path

import pytest

from sigmak import filings_db


def test_get_identifiers_with_filing_year_schema(tmp_path: Path):
    db_path = str(tmp_path / "with_year.db")

    # Ensure schema with filing_year exists via helper
    filings_db._ensure_db(db_path)

    # Insert two filings for same year; later date should be selected
    filings_db.insert_filing(db_path, "TEST", 2025, "ACC-1", "000000001", "https://sec/1", "2025-01-01")
    filings_db.insert_filing(db_path, "TEST", 2025, "ACC-2", "000000002", "https://sec/2", "2025-12-31")

    ids = filings_db.get_identifiers(db_path, "TEST", 2025)

    assert ids["accession"] == "ACC-2"
    assert ids["cik"] == "000000002"
    assert ids["sec_url"] == "https://sec/2"


def test_get_identifiers_with_filing_date_only_schema(tmp_path: Path):
    db_path = str(tmp_path / "date_only.db")

    # Create alternate schema like downloader's sec_filings.db (no filing_year column)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE filings_index (
            id TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            cik TEXT NOT NULL,
            accession TEXT NOT NULL,
            filing_type TEXT NOT NULL,
            filing_date DATE NOT NULL,
            sec_url TEXT NOT NULL,
            source TEXT NOT NULL,
            discovered_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            raw_metadata TEXT
        )
        """
    )
    # Insert two rows where filing_date year matches 2025; choose later date to be selected
    cur.execute(
        "INSERT INTO filings_index (id,ticker,cik,accession,filing_type,filing_date,sec_url,source) VALUES (?,?,?,?,?,?,?,?)",
        ("id1", "TEST", "000000010", "ACC-A", "10-K", "2025-01-01", "https://sec/a", "scrape"),
    )
    cur.execute(
        "INSERT INTO filings_index (id,ticker,cik,accession,filing_type,filing_date,sec_url,source) VALUES (?,?,?,?,?,?,?,?)",
        ("id2", "TEST", "000000011", "ACC-B", "10-K", "2025-10-31", "https://sec/b", "scrape"),
    )
    conn.commit()
    conn.close()

    ids = filings_db.get_identifiers(db_path, "TEST", 2025)
    assert ids["accession"] == "ACC-B"
    assert ids["cik"] == "000000011"
    assert ids["sec_url"] == "https://sec/b"
