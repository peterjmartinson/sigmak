# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

from typing import List, Dict, Any
import re


def is_boilerplate_intro(text: str) -> bool:
    """
    Detect if a chunk is the Item 1A title/intro boilerplate OR contains
    table of contents markers or non-risk financial data.
    
    Returns True if the text contains:
    - "Item 1A" followed by "Risk Factors" (case-insensitive)
    - "Table of Contents" markers
    - Financial/operational data sections (Operating Earnings, Gross profit, etc.)
    - Less than 50 words AND generic intro phrases
    
    Args:
        text: The chunk text to analyze
        
    Returns:
        bool: True if this appears to be boilerplate/non-risk content, False otherwise
    """
    text_lower = text.lower()
    word_count = len(text.split())
    
    # Filter out chunks with "Table of Contents" markers - these are navigation, not content
    if "table of contents" in text_lower:
        return True
    
    # Filter out chunks that contain financial/operational section headers (not risk factors)
    # These are section headers for financial discussion, not risk factor descriptions
    # Be very specific to avoid filtering legitimate risk content
    non_risk_sections = [
        "operating earnings",
        "gross profit margin",
        "consolidated statement",
        "balance sheet"
    ]
    
    # Only filter if the chunk contains these phrases AND lacks risk-related context
    # (e.g., a chunk about "Operating Earnings" is financial data, not a risk factor)
    if any(phrase in text_lower for phrase in non_risk_sections):
        # Double-check: if it also contains strong risk language, keep it
        risk_verbs = [
            "could adversely", "may adversely", "could negatively",
            "may negatively", "could harm", "may harm", "could damage",
            "may damage", "could impact", "may impact", "could affect",
            "risk that", "risk of", "risks", "threatening"
        ]
        # If it has non-risk section headers but NO risk language, filter it
        if not any(verb in text_lower for verb in risk_verbs):
            return True
    
    # Filter chunks that are just page numbers or section references
    # Example: "Item 1A. Risk Factors. 29"
    if word_count < 15 and re.search(r"item\s+1a.*\d+\s*$", text_lower):
        return True
    
    # Never filter chunks that contain substantive risk-related content
    # (even if they also contain the Item 1A header)
    substantive_keywords = [
        "risk", "competition", "market", "revenue", "financial", "regulatory",
        "operational", "customer", "supplier", "employee", "technology",
        "cyber", "litigation", "compliance", "debt", "liquidity",
        "inflation", "recession", "volatility", "disruption", "geopolitical"
    ]
    # Only check keywords for very short chunks (< 50 words)
    if word_count < 50 and any(keyword in text_lower for keyword in substantive_keywords):
        return False
    
    # Only filter very short chunks that are just section markers
    if word_count < 20 and re.search(r"item\s+1a", text_lower):
        return True
    
    # For longer chunks, only filter if they're BOTH:
    # 1. Short (< 50 words)
    # 2. Contain Item 1A header AND generic intro phrases  
    if word_count < 50:
        has_item_1a = re.search(r"item\s+1a[.\s\-:]+risk\s+factors", text_lower)
        if has_item_1a:
            intro_phrases = [
                "in this section, we describe",
                "the following describes",
                "following risk factors",
                "principal risks we face"
            ]
            # Only filter if it has any of these very specific intro templates
            if any(phrase in text_lower for phrase in intro_phrases):
                return True
    
    return False


def chunk_risk_section(text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Atomic function: Takes raw text and returns a list of
    chunked dictionaries with metadata attached.
    
    Now filters out boilerplate intro chunks to prevent them
    from appearing in risk analysis results.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = [
        {"text": chunk, "metadata": metadata}
        for chunk in splitter.split_text(text)
        if not is_boilerplate_intro(chunk)  # Filter out boilerplate
    ]
    
    return chunks
