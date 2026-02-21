# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def _strip_item_1a_header(text: str) -> str:
    """
    Remove Item 1A title line only.
    
    Let LLM classification handle detection of intro paragraphs and 
    other boilerplate text - it's better at semantic understanding.
    
    Args:
        text: Raw Item 1A text
        
    Returns:
        Text with title line stripped
    """
    return re.sub(
        r'^ITEM\s+1A[\.\s\-:]+RISK\s+FACTORS\s*\n?', 
        '', 
        text, 
        flags=re.IGNORECASE
    ).strip()


def is_valid_risk_chunk(text: str) -> bool:
    """
    Basic sanity check: Does this look like prose text?
    
    Detailed boilerplate detection is handled by LLM classification.
    This only catches obvious non-text garbage before embedding.
    
    Args:
        text: Risk chunk text to validate
        
    Returns:
        True if text appears to be readable prose
    """
    if not text or not text.strip():
        return False
    
    # Minimum viable content (not just a header)
    if len(text.split()) < 30:
        return False
    
    # Has sentence structure (not just keywords)
    if not re.search(r'[.!?]', text):
        return False
    
    # Not all-caps screaming (likely a section header)
    if text.isupper() and len(text) > 100:
        return False
    
    return True


def chunk_risk_section(text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Atomic function: Takes raw text and returns a list of
    chunked dictionaries with metadata attached.
    
    Strips Item 1A header and intro boilerplate before chunking to prevent
    generic intro text from being indexed as a risk factor.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # Strip boilerplate header/intro BEFORE chunking
    text = _strip_item_1a_header(text)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    return [
        {"text": chunk, "metadata": metadata}
        for chunk in splitter.split_text(text)
    ]
