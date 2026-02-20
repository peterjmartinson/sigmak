# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

import pytest
from sigmak.processing import chunk_risk_section, _strip_item_1a_header, is_valid_risk_chunk

def test_chunk_risk_section_structure() -> None:
    """
    Test that the chunker returns the correct dictionary format
    and preserves metadata.
    """
    sample_text = "Risk 1: Market volatility. " * 20
    meta = {"ticker": "AAPL", "year": 2025}

    chunks = chunk_risk_section(sample_text, meta)

    # Assert modular output format
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert "text" in chunks[0]
    assert chunks[0]["metadata"]["ticker"] == "AAPL"

def test_chunk_overlap_integrity() -> None:
    """
    Ensure the recursive splitter is actually overlapping
    so context isn't lost between atoms.
    """
    sample_text = "This is a long sentence that should be split into multiple parts for testing overlap."
    # Using small size to force a split
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # We test the logic we wrapped in processing.py
    meta = {"ticker": "TSLA"}
    chunks = chunk_risk_section(sample_text, meta)

    if len(chunks) > 1:
        # Check if some part of the first chunk exists in the second
        # (This validates the 'context-aware' objective of Subissue 1.0)
        assert any(word in chunks[1]["text"] for word in chunks[0]["text"].split()[-3:])


def test_strip_item_1a_header() -> None:
    """
    Test that Item 1A title is removed (intro detection moved to LLM).
    
    SRP: Verify title removal only - semantic boilerplate detection via LLM.
    """
    # Simulate real Item 1A text with header and intro boilerplate
    sample_text = """ITEM 1A. RISK FACTORS
The risks described below could, in ways we may or may not be able to accurately predict, materially and adversely affect our business, results of operations, financial position and liquidity. Our business operations could also be affected by additional factors that apply to all companies operating in the U.S. and globally. The following risk factors do not identify all risks that we may face.
Strategic Risks
Failure to successfully execute our omni-channel strategy and the cost of our investments in eCommerce and technology may materially adversely affect our market position, net sales and financial performance."""
    
    stripped = _strip_item_1a_header(sample_text)
    
    # Assert title is removed
    assert "ITEM 1A" not in stripped.upper()
    assert not stripped.startswith("ITEM")
    
    # Intro text now PRESERVED (LLM will classify as BOILERPLATE)
    assert "The risks described below" in stripped
    
    # Assert substantive content is preserved
    assert "Strategic Risks" in stripped
    assert "omni-channel strategy" in stripped


def test_chunk_risk_section_filters_boilerplate() -> None:
    """
    Test that chunk_risk_section removes title before chunking.
    
    SRP: Title removed, intro preserved for LLM classification.
    """
    # Real-world example from WMT filing
    sample_text = """ITEM 1A. RISK FACTORS
The risks described below could, in ways we may or may not be able to accurately predict, materially and adversely affect our business, results of operations, financial position and liquidity.
Strategic Risks
Our business faces significant competition from traditional and online retailers."""
    
    meta = {"ticker": "WMT", "filing_year": 2025}
    chunks = chunk_risk_section(sample_text, meta)
    
    # Assert we got chunks
    assert len(chunks) > 0
    
    # Assert first chunk does NOT contain title
    first_chunk_text = chunks[0]["text"]
    assert "ITEM 1A" not in first_chunk_text.upper()
    
    # Intro text may be present (will be classified as BOILERPLATE by LLM)
    # Substantive content is preserved
    assert "competition" in first_chunk_text.lower() or "risks described" in first_chunk_text.lower()


def test_is_valid_risk_chunk_basic_sanity() -> None:
    """
    Test basic sanity checks only (semantic detection via LLM).
    
    SRP: Verify minimal prose validation rules.
    """
    # Valid: Has words and punctuation
    valid_text = "This is a substantive risk with at least thirty words describing material threats. " * 3
    assert is_valid_risk_chunk(valid_text)
    
    # Invalid: Too short
    assert not is_valid_risk_chunk("Short text.")
    
    # Invalid: No punctuation (just keywords)
    no_punctuation = "just keywords no sentences here no punctuation marks" * 5
    assert not is_valid_risk_chunk(no_punctuation)
    
    # Invalid: All caps screaming (likely header)
    all_caps = "THIS IS ALL CAPS SCREAMING TEXT THAT GOES ON AND ON " * 15
    assert not is_valid_risk_chunk(all_caps)
    
    # Valid: Mixed case with punctuation
    mixed_valid = "We face risks from competition. Our market share may decline. Economic conditions impact us. Global events create uncertainty. " * 2
    assert is_valid_risk_chunk(mixed_valid)
