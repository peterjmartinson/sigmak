import json
from pathlib import Path

import pytest

from sigmak.integration import IntegrationPipeline, RiskAnalysisResult


def test_analyze_filing_uses_cached_output(tmp_path, monkeypatch):
    # Prepare dummy HTML file required by _validate_inputs
    html = tmp_path / "dummy_10k.html"
    html.write_text("<html><body>Item 1A dummy</body></html>")

    # Prepare cached output JSON
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    ticker = "TESTCO"
    year = 2020
    cache_file = output_dir / f"results_{ticker}_{year}.json"
    cached = {
        "ticker": ticker,
        "filing_year": year,
        "risks": [{"text": "cached risk", "severity": {"value": 0.5}, "novelty": {"value": 0.2}}],
        "metadata": {"cached": True}
    }
    cache_file.write_text(json.dumps(cached))

    pipeline = IntegrationPipeline(persist_path="./chroma_db_test")

    # Ensure indexing would raise if called (so test fails if indexing runs)
    def fail_index(*args, **kwargs):
        raise RuntimeError("indexing_called")

    monkeypatch.setattr(pipeline.indexing_pipeline, "index_filing", fail_index)

    # Call analyze_filing pointed at our dummy html; should load cached result
    result = pipeline.analyze_filing(str(html), ticker, year)

    assert isinstance(result, RiskAnalysisResult)
    assert result.ticker == ticker
    assert result.filing_year == year
    assert result.metadata.get("cached", False) is True
    assert len(result.risks) == 1
    assert result.risks[0]["text"] == "cached risk"
