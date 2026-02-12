# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from sigmak.ingest import (extract_text_from_file, parse_sec_html,
                                 slice_risk_factors)


# 1. Pure Atomic Logic Tests (In-Memory)
def test_parse_sec_html_removes_scripts() -> None:
    html = "<html><body><script>alert('bad');</script>Target Text</body></html>"
    result = parse_sec_html(html)
    assert "alert" not in result
    assert "Target Text" in result

def test_parse_sec_html_separates_tags() -> None:
    # Tests that <div>s don't result in mashedwords
    html = "<div>Word1</div><div>Word2</div>"
    result = parse_sec_html(html)
    assert result == "Word1 Word2"

# 2. IO / Integration Test (Using tmp_path)
def test_extract_text_from_file_handles_encoding(tmp_path) -> None:
    # Create a file with a non-UTF-8 character
    p = tmp_path / "legacy.html"
    # Writing a character that might trigger encoding issues
    p.write_bytes("<html><body>Item 1A ©</body></html>".encode('cp1252'))

    result = extract_text_from_file(p)
    assert "Item 1A" in result

def test_slice_risk_factors_isolates_content() -> None:
    sample = "ITEM 1. BUSINESS... ITEM 1A. RISK FACTORS. This is the risk. ITEM 2. PROPERTIES..."
    sliced = slice_risk_factors(sample)

    assert "BUSINESS" not in sliced.upper()
    assert "RISK FACTORS" in sliced.upper()
    assert "PROPERTIES" not in sliced.upper()

def test_slice_risk_factors_fallback_on_no_match() -> None:
    """
    Onion Check: Ensure function returns full text if 'Item 1A' marker is missing.
    """
    garbage_text = "This is a random document with no specific SEC markers."

    # We expect the function to return the original text untouched
    result = slice_risk_factors(garbage_text)

    assert result == garbage_text
    # Optional: If you use pytest's caplog, you can even verify a warning was logged

def test_slice_risk_factors_skips_toc_entry() -> None:
    """
    Critical: Verify extraction skips TOC and captures actual content section.

    Reproduces the bug where TOC entries like "Item 1A. Risk Factors 14" were
    being extracted instead of the actual multi-thousand word risk disclosure.
    """
    # Simulate a filing with TOC entry followed by actual section
    sample_text = """
    TABLE OF CONTENTS
    ITEM 1. BUSINESS 10
    ITEM 1A. RISK FACTORS 14
    ITEM 1B. UNRESOLVED STAFF COMMENTS 45

    [... lots of other content ...]

    ITEM 1. BUSINESS
    We design and manufacture electric vehicles.

    ITEM 1A. RISK FACTORS
    Investing in our common stock involves risks. Our business operations face
    significant challenges related to supply chain disruptions, manufacturing
    delays, regulatory compliance, and competitive pressures. The COVID-19
    pandemic has adversely impacted our operations and financial results.
    We rely heavily on international suppliers for critical battery components.
    Any disruption to these supply chains could materially harm our business.
    [... many more paragraphs of actual risk content ...]

    ITEM 1B. UNRESOLVED STAFF COMMENTS
    None.
    """

    sliced = slice_risk_factors(sample_text)

    # Should NOT be just a TOC reference
    assert "ITEM 1A. RISK FACTORS 14" != sliced.strip()

    # Should have substantial content (>500 chars minimum)
    assert len(sliced) > 500, f"Expected substantial content, got {len(sliced)} chars"

    # Should contain actual risk prose, not just headings
    assert "supply chain" in sliced.lower()
    assert "COVID-19" in sliced or "pandemic" in sliced.lower()
    assert "common stock involves risks" in sliced.lower()

    # Should NOT include the TOC section
    assert "TABLE OF CONTENTS" not in sliced

    # Should NOT include Item 1B content
    assert "UNRESOLVED STAFF COMMENTS" not in sliced or sliced.index("UNRESOLVED") > sliced.index("RISK FACTORS")


# =============================================================================
# Tests for edgartools integration (validation and extraction)
# =============================================================================

def test_validate_risk_factors_text_min_words() -> None:
    """
    Verify validation rejects content with fewer than minimum words (default 200).
    """
    from sigmak.ingest import validate_risk_factors_text
    from sigmak.config import EdgarConfig, EdgarValidationConfig
    
    validation = EdgarValidationConfig(
        min_words=200,
        max_words=50000,
        min_sentences=5,
        must_contain_risk=True
    )
    config = EdgarConfig(
        enabled=True,
        identity_email="test@example.com",
        validation=validation
    )
    
    # Create text with < 200 words (exactly 50 words)
    short_text = "risk " * 50  # 50 words, contains "risk"
    
    assert not validate_risk_factors_text(short_text, config)


def test_validate_risk_factors_text_max_words() -> None:
    """
    Verify validation rejects content exceeding maximum words (default 50,000).
    """
    from sigmak.ingest import validate_risk_factors_text
    from sigmak.config import EdgarConfig, EdgarValidationConfig
    
    validation = EdgarValidationConfig(
        min_words=200,
        max_words=50000,
        min_sentences=5,
        must_contain_risk=True
    )
    config = EdgarConfig(
        enabled=True,
        identity_email="test@example.com",
        validation=validation
    )
    
    # Create text with > 50,000 words
    too_long_text = "risk word. " * 60000  # 120,000 words
    
    assert not validate_risk_factors_text(too_long_text, config)


def test_validate_risk_factors_text_must_contain_risk() -> None:
    """
    Verify validation rejects content without the word 'risk' when required.
    """
    from sigmak.ingest import validate_risk_factors_text
    from sigmak.config import EdgarConfig, EdgarValidationConfig
    
    validation = EdgarValidationConfig(
        min_words=200,
        max_words=50000,
        min_sentences=5,
        must_contain_risk=True
    )
    config = EdgarConfig(
        enabled=True,
        identity_email="test@example.com",
        validation=validation
    )
    
    # 250 words, 10 sentences, but no "risk"
    no_risk_text = "The company operates globally. " * 10 + ("Business operations continue smoothly. " * 60)
    
    assert not validate_risk_factors_text(no_risk_text, config)


def test_validate_risk_factors_text_toc_pattern() -> None:
    """
    Verify validation rejects TOC-like entries (e.g., 'Risk Factors...14').
    """
    from sigmak.ingest import validate_risk_factors_text
    from sigmak.config import EdgarConfig, EdgarValidationConfig
    
    validation = EdgarValidationConfig(
        min_words=200,
        max_words=50000,
        min_sentences=5,
        must_contain_risk=True
    )
    config = EdgarConfig(
        enabled=True,
        identity_email="test@example.com",
        validation=validation
    )
    
    # TOC-style entries should be rejected
    toc_entry1 = "Item 1A. Risk Factors...........................14"
    toc_entry2 = "RISK FACTORS 23"
    toc_entry3 = "Item 1A. Risk Factors Page 45"
    
    assert not validate_risk_factors_text(toc_entry1, config)
    assert not validate_risk_factors_text(toc_entry2, config)
    assert not validate_risk_factors_text(toc_entry3, config)


def test_validate_risk_factors_text_min_sentences() -> None:
    """
    Verify validation rejects content with fewer than minimum sentences (default 5).
    """
    from sigmak.ingest import validate_risk_factors_text
    from sigmak.config import EdgarConfig, EdgarValidationConfig
    
    validation = EdgarValidationConfig(
        min_words=200,
        max_words=50000,
        min_sentences=5,
        must_contain_risk=True
    )
    config = EdgarConfig(
        enabled=True,
        identity_email="test@example.com",
        validation=validation
    )
    
    # 250 words but only 2 sentences
    few_sentences = "Risk factors include many things and lots of words here. " + ("word " * 200)
    
    assert not validate_risk_factors_text(few_sentences, config)


def test_validate_risk_factors_text_valid_content() -> None:
    """
    Verify validation accepts valid Item 1A risk factor text.
    """
    from sigmak.ingest import validate_risk_factors_text
    from sigmak.config import EdgarConfig, EdgarValidationConfig
    
    validation = EdgarValidationConfig(
        min_words=200,
        max_words=50000,
        min_sentences=5,
        must_contain_risk=True
    )
    config = EdgarConfig(
        enabled=True,
        identity_email="test@example.com",
        validation=validation
    )
    
    # Valid risk factors text: 250+ words, 10+ sentences, contains "risk"
    valid_text = """
    Investing in our common stock involves substantial risks. Our business operations 
    face significant challenges related to supply chain disruptions. Manufacturing 
    delays pose considerable risks to our revenue targets. Regulatory compliance 
    requirements continue to evolve and create uncertainty. Competitive pressures 
    in the market may adversely affect our margins. We rely heavily on international 
    suppliers for critical components. Any disruption to these supply chains could 
    materially harm our business operations. The economic environment remains 
    challenging and unpredictable. Currency fluctuations may impact our financial 
    results significantly. Cybersecurity threats pose ongoing risks to our systems.
    """ + (" Additional risk factors include various operational challenges. " * 30)
    
    assert validate_risk_factors_text(valid_text, config)


def test_extract_risk_factors_via_edgartools_success() -> None:
    """
    Verify edgartools extraction succeeds when properly configured and API responds.
    """
    from sigmak.ingest import extract_risk_factors_via_edgartools
    from sigmak.config import EdgarConfig, EdgarValidationConfig
    
    validation = EdgarValidationConfig(
        min_words=200,
        max_words=50000,
        min_sentences=5,
        must_contain_risk=True
    )
    config = EdgarConfig(
        enabled=True,
        identity_email="test@example.com",
        validation=validation
    )
    
    # Mock the edgartools library
    mock_risk_text = """
    Investing in our securities involves substantial risks. Our business faces 
    challenges from market competition. Supply chain disruptions may harm operations.
    Regulatory changes could increase compliance costs. Economic downturns affect demand.
    Technology failures pose operational risks. International expansion carries risks.
    Currency fluctuations impact financial results. Cybersecurity threats are ongoing.
    Legal proceedings may result in significant costs. Climate change poses long-term risks.
    """ + (" Additional detailed risk factors and considerations. " * 50)
    
    mock_tenk = MagicMock()
    mock_tenk.risk_factors = mock_risk_text
    
    mock_filing = MagicMock()
    mock_filing.obj.return_value = mock_tenk
    
    mock_filings = MagicMock()
    mock_filings.latest.return_value = mock_filing
    
    mock_company = MagicMock()
    mock_company.get_filings.return_value = mock_filings
    
    # Patch edgartools imports and function calls
    with patch("edgar.Company", return_value=mock_company):
        with patch("edgar.set_identity") as mock_set_identity:
            result = extract_risk_factors_via_edgartools("AAPL", 2024, config)
    
    assert result is not None
    assert "risks" in result.lower()
    assert len(result.split()) > 200
    mock_set_identity.assert_called_once_with(config.identity_email)


def test_extract_risk_factors_via_edgartools_not_installed() -> None:
    """
    Verify graceful handling when edgartools is not installed.
    """
    from sigmak.ingest import extract_risk_factors_via_edgartools
    from sigmak.config import EdgarConfig, EdgarValidationConfig
    
    validation = EdgarValidationConfig(
        min_words=200,
        max_words=50000,
        min_sentences=5,
        must_contain_risk=True
    )
    config = EdgarConfig(
        enabled=True,
        identity_email="test@example.com",
        validation=validation
    )
    
    # Simulate ImportError for edgartools
    with patch("edgar.Company", side_effect=ImportError("No module named 'edgar'")):
        result = extract_risk_factors_via_edgartools("AAPL", 2024, config)
    
    assert result is None


def test_extract_risk_factors_via_edgartools_api_error() -> None:
    """
    Verify graceful handling when edgartools API fails.
    """
    from sigmak.ingest import extract_risk_factors_via_edgartools
    from sigmak.config import EdgarConfig, EdgarValidationConfig
    
    validation = EdgarValidationConfig(
        min_words=200,
        max_words=50000,
        min_sentences=5,
        must_contain_risk=True
    )
    config = EdgarConfig(
        enabled=True,
        identity_email="test@example.com",
        validation=validation
    )
    
    # Mock API failure
    mock_company = MagicMock()
    mock_company.get_filings.side_effect = Exception("SEC API error")
    
    with patch("edgar.Company", return_value=mock_company):
        with patch("edgar.set_identity"):
            result = extract_risk_factors_via_edgartools("AAPL", 2024, config)
    
    assert result is None


def test_extract_risk_factors_with_fallback_prefers_edgartools(tmp_path) -> None:
    """
    Verify orchestrator tries edgartools first when enabled and available.
    """
    from sigmak.ingest import extract_risk_factors_with_fallback
    from sigmak.config import Config, EdgarConfig, EdgarValidationConfig, LoggingConfig, DatabaseConfig, ChromaConfig, LLMConfig, DriftConfig
    from pathlib import Path
    
    # Create mock HTML file for fallback
    html_file = tmp_path / "test_10k.html"
    html_file.write_text("""
        <html><body>
        ITEM 1A. RISK FACTORS
        Fallback risk text from HTML file parsing.
        This is what we get from the regex-based extraction.
        ITEM 1B. UNRESOLVED STAFF COMMENTS
        </body></html>
    """)
    
    # Create mock edgartools response (valid)
    mock_risk_text = """
    Edgartools extracted risk factors. Our business faces substantial risks.
    Market competition poses challenges. Supply chain issues may harm operations.
    Regulatory changes could increase costs. Economic downturns affect demand.
    Technology failures pose operational risks. Currency fluctuations impact results.
    Cybersecurity threats are ongoing. Legal proceedings cost money. Climate risks exist.
    """ + (" More risk details from edgartools API. " * 50)
    
    mock_tenk = MagicMock()
    mock_tenk.risk_factors = mock_risk_text
    
    mock_filing = MagicMock()
    mock_filing.obj.return_value = mock_tenk
    
    mock_filings = MagicMock()
    mock_filings.latest.return_value = mock_filing
    
    mock_company = MagicMock()
    mock_company.get_filings.return_value = mock_filings
    
    # Create settings with edgartools enabled
    validation = EdgarValidationConfig(
        min_words=200,
        max_words=50000,
        min_sentences=5,
        must_contain_risk=True
    )
    edgar = EdgarConfig(
        enabled=True,
        identity_email="test@example.com",
        validation=validation
    )
    config = Config(
        database=DatabaseConfig(sqlite_path=Path("test.db")),
        chroma=ChromaConfig(persist_directory=Path("test_db"), embedding_model="test", llm_cache_similarity_threshold=0.8),
        llm=LLMConfig(model="test", temperature=0.0),
        drift=DriftConfig(review_cron="0 3 * * *", sample_size=100, low_confidence_threshold=0.6, drift_threshold=0.2),
        logging=LoggingConfig(level="INFO"),
        edgar=edgar,
        redis_url="redis://localhost",
        environment="test"
    )
    
    with patch("edgar.Company", return_value=mock_company):
        with patch("edgar.set_identity"):
            text, method = extract_risk_factors_with_fallback("AAPL", 2024, str(html_file), config)
    
    assert method == "edgartools"
    assert "edgartools" in text.lower()
    assert "fallback" not in text.lower()


def test_extract_risk_factors_with_fallback_uses_fallback(tmp_path) -> None:
    """
    Verify orchestrator falls back to regex extraction when edgartools fails.
    """
    from sigmak.ingest import extract_risk_factors_with_fallback
    from sigmak.config import Config, EdgarConfig, EdgarValidationConfig, LoggingConfig, DatabaseConfig, ChromaConfig, LLMConfig, DriftConfig
    from pathlib import Path
    
    # Create HTML file for fallback
    html_file = tmp_path / "test_10k.html"
    html_file.write_text("""
        <html><body>
        ITEM 1A. RISK FACTORS
        Fallback risk extraction works correctly. Our business operations involve risks.
        We face significant market challenges and competitive pressures daily.
        Supply chain disruptions could materially harm our business operations.
        Regulatory compliance costs continue to increase over time significantly.
        Economic conditions may adversely affect our financial performance.
        Technology infrastructure failures pose operational risks to systems.
        International expansion carries inherent risks and uncertainties.
        Currency exchange rate fluctuations impact our reported results.
        Cybersecurity threats require ongoing investment and vigilance.
        Climate change may affect our long-term business operations.
        ITEM 1B. UNRESOLVED STAFF COMMENTS
        </body></html>
    """)
    
    # Create settings with edgartools enabled but mock it to fail
    validation = EdgarValidationConfig(
        min_words=200,
        max_words=50000,
        min_sentences=5,
        must_contain_risk=True
    )
    edgar = EdgarConfig(
        enabled=True,
        identity_email="test@example.com",
        validation=validation
    )
    config = Config(
        database=DatabaseConfig(sqlite_path=Path("test.db")),
        chroma=ChromaConfig(persist_directory=Path("test_db"), embedding_model="test", llm_cache_similarity_threshold=0.8),
        llm=LLMConfig(model="test", temperature=0.0),
        drift=DriftConfig(review_cron="0 3 * * *", sample_size=100, low_confidence_threshold=0.6, drift_threshold=0.2),
        logging=LoggingConfig(level="INFO"),
        edgar=edgar,
        redis_url="redis://localhost",
        environment="test"
    )
    
    # Mock edgartools to fail
    with patch("edgar.Company", side_effect=ImportError("No edgartools")):
        text, method = extract_risk_factors_with_fallback("AAPL", 2024, str(html_file), config)
    
    assert method == "fallback"
    assert "fallback" in text.lower()
    assert "business operations involve risks" in text.lower()


def test_extract_risk_factors_with_fallback_disabled(tmp_path) -> None:
    """
    Verify orchestrator skips edgartools when disabled in config.
    """
    from sigmak.ingest import extract_risk_factors_with_fallback
    from sigmak.config import Config, EdgarConfig, EdgarValidationConfig, LoggingConfig, DatabaseConfig, ChromaConfig, LLMConfig, DriftConfig
    from pathlib import Path
    
    # Create HTML file for fallback
    html_file = tmp_path / "test_10k.html"
    html_file.write_text("""
        <html><body>
        ITEM 1A. RISK FACTORS
        Risk factors extracted via fallback mechanism when edgartools disabled.
        Our company faces various business and operational risks continuously.
        Market conditions may adversely affect our financial performance metrics.
        Competitive pressures require ongoing strategic responses and investments.
        Supply chain dependencies create vulnerabilities in our operations.
        Regulatory requirements impose significant compliance costs and burdens.
        Technology disruptions could harm our business operations materially.
        Economic downturns reduce demand for our products and services.
        Currency fluctuations affect international revenue and cost structures.
        Cybersecurity incidents pose threats to our data and systems.
        Climate risks may impact long-term business sustainability goals.
        ITEM 1B. UNRESOLVED STAFF COMMENTS
        </body></html>
    """)
    
    # Create settings with edgartools DISABLED
    validation = EdgarValidationConfig(
        min_words=200,
        max_words=50000,
        min_sentences=5,
        must_contain_risk=True
    )
    edgar = EdgarConfig(
        enabled=False,  # Disabled
        identity_email="test@example.com",
        validation=validation
    )
    config = Config(
        database=DatabaseConfig(sqlite_path=Path("test.db")),
        chroma=ChromaConfig(persist_directory=Path("test_db"), embedding_model="test", llm_cache_similarity_threshold=0.8),
        llm=LLMConfig(model="test", temperature=0.0),
        drift=DriftConfig(review_cron="0 3 * * *", sample_size=100, low_confidence_threshold=0.6, drift_threshold=0.2),
        logging=LoggingConfig(level="INFO"),
        edgar=edgar,
        redis_url="redis://localhost",
        environment="test"
    )
    
    # Should go straight to fallback without trying edgartools
    text, method = extract_risk_factors_with_fallback("AAPL", 2024, str(html_file), config)
    
    assert method == "fallback"
    assert "risk factors extracted" in text.lower()
