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
    from sigmak.config import DomExtractorConfig, DomExtractorValidationConfig
    
    validation = DomExtractorValidationConfig(
        min_words=200,
        max_words=50000,
        min_sentences=5,
        must_contain_risk=True
    )
    config = DomExtractorConfig(
        enabled=True,
        method="sec-parser",
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
    from sigmak.config import DomExtractorConfig, DomExtractorValidationConfig
    
    validation = DomExtractorValidationConfig(
        min_words=200,
        max_words=50000,
        min_sentences=5,
        must_contain_risk=True
    )
    config = DomExtractorConfig(
        enabled=True,
        method="sec-parser",
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
    from sigmak.config import DomExtractorConfig, DomExtractorValidationConfig
    
    validation = DomExtractorValidationConfig(
        min_words=200,
        max_words=50000,
        min_sentences=5,
        must_contain_risk=True
    )
    config = DomExtractorConfig(
        enabled=True,
        method="sec-parser",
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
    from sigmak.config import DomExtractorConfig, DomExtractorValidationConfig
    
    validation = DomExtractorValidationConfig(
        min_words=200,
        max_words=50000,
        min_sentences=5,
        must_contain_risk=True
    )
    config = DomExtractorConfig(
        enabled=True,
        method="sec-parser",
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
    from sigmak.config import DomExtractorConfig, DomExtractorValidationConfig
    
    validation = DomExtractorValidationConfig(
        min_words=200,
        max_words=50000,
        min_sentences=5,
        must_contain_risk=True
    )
    config = DomExtractorConfig(
        enabled=True,
        method="sec-parser",
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
    from sigmak.config import DomExtractorConfig, DomExtractorValidationConfig
    
    validation = DomExtractorValidationConfig(
        min_words=200,
        max_words=50000,
        min_sentences=5,
        must_contain_risk=True
    )
    config = DomExtractorConfig(
        enabled=True,
        method="sec-parser",
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


def test_extract_risk_factors_via_secparser_success(tmp_path) -> None:
    """
    Verify sec-parser extraction succeeds with valid HTML file.
    """
    from sigmak.ingest import extract_risk_factors_via_secparser
    from sigmak.config import DomExtractorConfig, DomExtractorValidationConfig
    
    validation = DomExtractorValidationConfig(
        min_words=200,
        max_words=50000,
        min_sentences=5,
        must_contain_risk=True
    )
    config = DomExtractorConfig(
        enabled=True,
        method="sec-parser",
        validation=validation
    )
    
    # Create mock HTML file with Item 1A section
    html_file = tmp_path / "test_10k.html"
    # Create substantial content that will pass validation (>200 words, >5 sentences, contains "risk")
    risk_content = """
    Investing in our securities involves substantial risks that could materially harm our business.
    Our business faces significant challenges from intense market competition in all regions.
    Supply chain disruptions may adversely harm our operations and financial results.
    Regulatory changes could substantially increase our compliance costs and operational burden.
    Economic downturns significantly affect customer demand for our products and services.
    Technology infrastructure failures pose critical operational risks to our systems.
    International expansion efforts carry inherent risks and substantial uncertainties.
    Currency exchange rate fluctuations materially impact our reported financial results.
    Cybersecurity threats require ongoing investment, vigilance, and proactive management.
    Legal proceedings may result in significant financial costs and reputational damage.
    Climate change poses long-term risks to our business operations and supply chain.
    Market volatility creates uncertainty in our revenue forecasting and planning processes.
    """
    additional_risks = risk_content * 5  # Repeat to ensure >200 words
    
    html_content = f"""
    <html>
        <body>
            <div>ITEM 1. BUSINESS</div>
            <div>Business description here.</div>
            <div>ITEM 1A. RISK FACTORS</div>
            <div>{additional_risks}</div>
            <div>ITEM 1B. UNRESOLVED STAFF COMMENTS</div>
        </body>
    </html>
    """
    html_file.write_text(html_content)
    
    result = extract_risk_factors_via_secparser(str(html_file), config)
    
    assert result is not None
    assert "risks" in result.lower()
    assert len(result.split()) > 200
    assert "ITEM 1B" not in result


def test_extract_risk_factors_via_secparser_not_installed() -> None:
    """
    Verify graceful handling when sec-parser is not installed.
    """
    from sigmak.ingest import extract_risk_factors_via_secparser
    from sigmak.config import DomExtractorConfig, DomExtractorValidationConfig
    import sys
    
    validation = DomExtractorValidationConfig(
        min_words=200,
        max_words=50000,
        min_sentences=5,
        must_contain_risk=True
    )
    config = DomExtractorConfig(
        enabled=True,
        method="sec-parser",
        validation=validation
    )
    
    # Temporarily remove sec_parser from sys.modules to simulate not installed
    sec_parser_backup = sys.modules.get('sec_parser')
    try:
        if 'sec_parser' in sys.modules:
            del sys.modules['sec_parser']
        
        # Mock __import__ to raise ImportError for sec_parser only
        import builtins
        original_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == 'sec_parser' or name.startswith('sec_parser.'):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            result = extract_risk_factors_via_secparser("/fake/path.html", config)
        
        assert result is None
    finally:
        # Restore sec_parser in sys.modules if it was there before
        if sec_parser_backup is not None:
            sys.modules['sec_parser'] = sec_parser_backup


def test_extract_risk_factors_via_secparser_parsing_error(tmp_path) -> None:
    """
    Verify graceful handling when sec-parser encounters parsing errors.
    """
    from sigmak.ingest import extract_risk_factors_via_secparser
    from sigmak.config import DomExtractorConfig, DomExtractorValidationConfig
    
    validation = DomExtractorValidationConfig(
        min_words=200,
        max_words=50000,
        min_sentences=5,
        must_contain_risk=True
    )
    config = DomExtractorConfig(
        enabled=True,
        method="sec-parser",
        validation=validation
    )
    
    # Create HTML file with no Item 1A section
    html_file = tmp_path / "bad_10k.html"
    html_file.write_text("<html><body>Invalid content without Item 1A</body></html>")
    
    result = extract_risk_factors_via_secparser(str(html_file), config)
    
    assert result is None


def test_extract_risk_factors_with_fallback_prefers_secparser(tmp_path) -> None:
    """
    Verify orchestrator tries sec-parser first when enabled and available.
    """
    from sigmak.ingest import extract_risk_factors_with_fallback
    from sigmak.config import Config, DomExtractorConfig, DomExtractorValidationConfig, LoggingConfig, DatabaseConfig, ChromaConfig, LLMConfig, DriftConfig
    from pathlib import Path
    
    # Create HTML file with Item 1A content
    html_file = tmp_path / "test_10k.html"
    # Create more realistic HTML that sec-parser can properly extract
    html_content = """
        <html><body>
        <div>ITEM 1. BUSINESS</div>
        <p>Business content goes here with details about operations.</p>
        
        <div>ITEM 1A. RISK FACTORS</div>
        <p>Our business faces substantial risks that could materially affect our operations.
        Market competition poses significant challenges to our market share and pricing.</p>
        <p>Supply chain disruptions and shortages may materially harm our operations and ability to deliver.
        Regulatory changes in multiple jurisdictions could significantly increase our compliance costs.</p>
        <p>Economic downturns and recessions affect customer demand and purchasing patterns.
        Technology infrastructure failures pose serious operational risks to our systems and data.</p>
        <p>Currency exchange rate fluctuations materially impact our international financial results.
        Cybersecurity threats and data breaches require ongoing investment and vigilance.</p>
        <p>Legal proceedings and litigation can be costly and time-consuming for our organization.
        Climate change and environmental regulations may affect our long-term business model.</p>
        <p>Geopolitical tensions and trade disputes create uncertainty in our global operations.
        Labor shortages and workforce challenges impact our ability to scale operations.</p>
        <p>Interest rate changes affect our borrowing costs and capital structure decisions.
        Product liability claims could result in significant financial losses and reputational damage.</p>
        <p>Intellectual property disputes may require substantial legal resources to defend.
        Changes in consumer preferences could reduce demand for our products and services.</p>
        <p>Pandemic or health crisis events may disrupt operations and supply chains globally.
        Insurance coverage may not be adequate to cover all potential losses and liabilities.</p>
        <p>Raw material price volatility affects our production costs and profit margins.
        Dependence on key suppliers creates concentration risk in our supply chain.</p>
        
        <div>ITEM 1B. UNRESOLVED STAFF COMMENTS</div>
        <p>We have no unresolved comments from the SEC staff.</p>
        </body></html>
    """
    html_file.write_text(html_content)
    
    # Create settings with dom_extractor enabled
    validation = DomExtractorValidationConfig(
        min_words=200,
        max_words=50000,
        min_sentences=5,
        must_contain_risk=True
    )
    dom_extractor = DomExtractorConfig(
        enabled=True,
        method="sec-parser",
        validation=validation
    )
    config = Config(
        database=DatabaseConfig(sqlite_path=Path("test.db")),
        chroma=ChromaConfig(persist_directory=Path("test_db"), embedding_model="test", llm_cache_similarity_threshold=0.8),
        llm=LLMConfig(model="test", temperature=0.0),
        drift=DriftConfig(review_cron="0 3 * * *", sample_size=100, low_confidence_threshold=0.6, drift_threshold=0.2),
        logging=LoggingConfig(level="INFO"),
        dom_extractor=dom_extractor,
        redis_url="redis://localhost",
        environment="test"
    )
    
    text, method = extract_risk_factors_with_fallback("AAPL", 2024, str(html_file), config)
    
    assert method == "sec-parser"
    assert "sec-parser" in text.lower() or "risk" in text.lower()
    assert len(text.split()) > 200


def test_extract_risk_factors_with_fallback_uses_regex(tmp_path) -> None:
    """
    Verify orchestrator falls back to regex extraction when sec-parser fails.
    """
    from sigmak.ingest import extract_risk_factors_with_fallback
    from sigmak.config import Config, DomExtractorConfig, DomExtractorValidationConfig, LoggingConfig, DatabaseConfig, ChromaConfig, LLMConfig, DriftConfig
    from pathlib import Path
    
    # Create HTML file for regex fallback
    html_file = tmp_path / "test_10k.html"
    html_file.write_text("""
        <html><body>
        ITEM 1A. RISK FACTORS
        Regex fallback extraction works correctly. Our business operations involve risks.
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
    
    # Create settings with dom_extractor enabled
    validation = DomExtractorValidationConfig(
        min_words=200,
        max_words=50000,
        min_sentences=5,
        must_contain_risk=True
    )
    dom_extractor = DomExtractorConfig(
        enabled=True,
        method="sec-parser",
        validation=validation
    )
    config = Config(
        database=DatabaseConfig(sqlite_path=Path("test.db")),
        chroma=ChromaConfig(persist_directory=Path("test_db"), embedding_model="test", llm_cache_similarity_threshold=0.8),
        llm=LLMConfig(model="test", temperature=0.0),
        drift=DriftConfig(review_cron="0 3 * * *", sample_size=100, low_confidence_threshold=0.6, drift_threshold=0.2),
        logging=LoggingConfig(level="INFO"),
        dom_extractor=dom_extractor,
        redis_url="redis://localhost",
        environment="test"
    )
    
    # Mock sec-parser to fail by raising ImportError
    # This will cause it to fall back to regex
    import builtins
    import sys
    
    sec_parser_backup = sys.modules.get('sec_parser')
    original_import = builtins.__import__
    
    def mock_import(name, *args, **kwargs):
        if name == 'sec_parser' or name.startswith('sec_parser.'):
            raise ImportError(f"No module named '{name}'")
        return original_import(name, *args, **kwargs)
    
    try:
        if 'sec_parser' in sys.modules:
            del sys.modules['sec_parser']
        
        builtins.__import__ = mock_import
        text, method = extract_risk_factors_with_fallback("AAPL", 2024, str(html_file), config)
    finally:
        builtins.__import__ = original_import
        if sec_parser_backup is not None:
            sys.modules['sec_parser'] = sec_parser_backup
    
    assert method == "regex"
    assert "business operations involve risks" in text.lower()


def test_extract_risk_factors_with_fallback_disabled(tmp_path) -> None:
    """
    Verify orchestrator skips sec-parser when disabled in config.
    """
    from sigmak.ingest import extract_risk_factors_with_fallback
    from sigmak.config import Config, DomExtractorConfig, DomExtractorValidationConfig, LoggingConfig, DatabaseConfig, ChromaConfig, LLMConfig, DriftConfig
    from pathlib import Path
    
    # Create HTML file for regex extraction
    html_file = tmp_path / "test_10k.html"
    html_file.write_text("""
        <html><body>
        ITEM 1A. RISK FACTORS
        Risk factors extracted via regex mechanism when dom_extractor disabled.
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
    
    # Create settings with dom_extractor DISABLED
    validation = DomExtractorValidationConfig(
        min_words=200,
        max_words=50000,
        min_sentences=5,
        must_contain_risk=True
    )
    dom_extractor = DomExtractorConfig(
        enabled=False,  # Disabled
        method="sec-parser",
        validation=validation
    )
    config = Config(
        database=DatabaseConfig(sqlite_path=Path("test.db")),
        chroma=ChromaConfig(persist_directory=Path("test_db"), embedding_model="test", llm_cache_similarity_threshold=0.8),
        llm=LLMConfig(model="test", temperature=0.0),
        drift=DriftConfig(review_cron="0 3 * * *", sample_size=100, low_confidence_threshold=0.6, drift_threshold=0.2),
        logging=LoggingConfig(level="INFO"),
        dom_extractor=dom_extractor,
        redis_url="redis://localhost",
        environment="test"
    )
    
    # Should go straight to regex without trying sec-parser
    text, method = extract_risk_factors_with_fallback("AAPL", 2024, str(html_file), config)
    
    assert method == "regex"
    assert "risk factors extracted" in text.lower()
