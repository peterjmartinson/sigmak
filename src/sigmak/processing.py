# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

from typing import List, Dict, Any
import re


def is_boilerplate_intro(text: str) -> bool:
    """
    Detect if a chunk is the Item 1A title/intro boilerplate.
    
    Returns True if the text contains:
    - "Item 1A" followed by "Risk Factors" (case-insensitive)
    - Less than 50 words (intro text is typically short and generic)
    - Contains phrases like "In this section", "described below", "following risks"
    
    Args:
        text: The chunk text to analyze
        
    Returns:
        bool: True if this appears to be boilerplate intro, False otherwise
    """
    text_lower = text.lower()
    word_count = len(text.split())
    
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
