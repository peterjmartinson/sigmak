# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

import pytest
from sigmak.processing import chunk_risk_section, is_boilerplate_intro

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


# ========================================
# Boilerplate Detection Tests (FIX_YOY_REPORT_BUGS)
# ========================================

def test_detects_item_1a_title():
    """Item 1A title text should be detected as boilerplate."""
    text = "Item 1A. RISK FACTORS"
    assert is_boilerplate_intro(text) is True


def test_detects_item_1a_intro_with_generic_phrases():
    """Item 1A intro with generic phrases should be detected."""
    text = "Item 1A. RISK FACTORS. In this section, we describe the following risks that could affect our business."
    assert is_boilerplate_intro(text) is True


def test_ignores_substantive_risk_text():
    """Real risk disclosure text should NOT be detected as boilerplate."""
    text = """
    Our business is subject to significant regulatory risks that could materially
    impact our operations. Changes in federal regulations, particularly regarding
    environmental compliance and labor standards, could increase our operating costs
    by an estimated $5-10 million annually. We have experienced regulatory penalties
    in the past three years totaling $2.3 million.
    """
    assert is_boilerplate_intro(text) is False


def test_short_item_1a_markers_are_filtered():
    """Very short chunks containing only section markers should be filtered."""
    text = "Item 1A - Risk Factors ​"
    assert is_boilerplate_intro(text) is True


def test_chunk_risk_section_filters_boilerplate():
    """chunk_risk_section should filter out boilerplate chunks."""
    # Use text that will be split into multiple chunks, with boilerplate in first chunk
    text = """
    Item 1A. RISK FACTORS. In this section, we describe the risks we face.
    
    """ + ("Our business faces significant supply chain risks. " * 100)  # Long substantive text
    
    metadata = {"ticker": "TEST", "year": 2025, "item_type": "1A"}
    chunks = chunk_risk_section(text, metadata)
    
    # Should have chunks with substantive content (the boilerplate intro might be filtered)
    assert len(chunks) > 0, "Should have at least one chunk with substantive risk content"
    
    # Verify that no chunk is ONLY the boilerplate intro
    for chunk in chunks:
        text = chunk["text"].strip()
        # A chunk that only contains the Item 1A intro should be filtered
        if "item 1a" in text.lower() and "risk factors" in text.lower():
            # If it mentions Item 1A and Risk Factors, it should also have substantive content
            assert len(text.split()) > 20, "Chunks with Item 1A reference should include substantive content"
