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
    
    # Pattern 1: Direct Item 1A title (most common)
    if re.search(r"item\s+1a[.\s\-:]+risk\s+factors", text_lower):
        # If it's short (< 50 words), definitely boilerplate
        if word_count < 50:
            return True
        # If it contains generic intro phrases, likely boilerplate
        intro_phrases = [
            "in this section",
            "described below",
            "following risks",
            "we are subject to",
            "includes the following",
            "set forth below"
        ]
        if any(phrase in text_lower for phrase in intro_phrases):
            return True
    
    # Pattern 2: Very short chunks that are just section markers
    if word_count < 15 and re.search(r"item\s+1a", text_lower):
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
