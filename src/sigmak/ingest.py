# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

import logging
import re
from pathlib import Path
from typing import Optional, Tuple

from bs4 import BeautifulSoup

from sigmak.config import Config, EdgarConfig
from sigmak.processing import chunk_risk_section

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_sec_html(html_content: str) -> str:
    """
    Pure logic: Extracts clean text from raw HTML content.
    Targeted for unit testing and modular use.
    """
    # Use lxml for speed and better handling of malformed SEC tags
    soup = BeautifulSoup(html_content, "lxml")

    # Remove script and style elements that contaminate RAG context
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()

    # get_text with a separator prevents words from mashing together
    # when <td> or <div> tags end.
    text: str = soup.get_text(separator=" ", strip=True)
    return text

def extract_text_from_file(html_path: str | Path) -> str:
    """
    IO Wrapper: Handles file loading, encoding fallbacks, and calls the parser.
    """
    path = Path(html_path)
    if not path.exists():
        logger.error(f"File not found: {html_path}")
        raise FileNotFoundError(f"No HTM file found at {html_path}")

    try:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Attempt 2: CP1252 (The legacy Windows fallback common in SEC filings)
            logger.warning(f"UTF-8 decode failed for {html_path}. Falling back to CP1252.")
            with open(path, 'r', encoding='cp1252') as f:
                content = f.read()

        full_text = parse_sec_html(content)

        logger.info(f"Successfully extracted {len(full_text)} characters from {html_path}")
        return full_text

    except Exception as e:
        logger.error(f"Failed to process HTM file {html_path}: {e}")
        raise

def slice_risk_factors(text: str) -> str:
    """
    Slices the 'Item 1A' section from the full text.
    Handles both TOC entries and actual section headers by selecting the match
    with the most substantive prose content following it.
    Gracefully falls back to full text if markers are missing.
    """
    # Regex to find Item 1A, ignoring case and handling varied spacing/punctuation
    start_pattern = re.compile(r"ITEM\s+1A[\.\s\-:]+RISK\s+FACTORS", re.IGNORECASE)

    # End markers: Item 1B or Item 2 are the most common next sections
    end_pattern = re.compile(r"ITEM\s+(1B|2)[\.\s\-:]", re.IGNORECASE)

    # Find ALL occurrences of Item 1A
    matches = list(start_pattern.finditer(text))
    if not matches:
        logger.warning("Item 1A marker not found. Fallback: returning full text.")
        return text

    # Strategy: Use the match with the longest text following it
    # The TOC entry will have minimal text, the real section will have thousands of words
    best_match = None
    max_length = 0

    for match in matches:
        # Look ahead 500 characters to see if this is substantive content
        sample = text[match.end():match.end() + 500]
        # Real risk sections have lots of prose, not just links/numbers/page refs
        word_count = len(sample.split())

        if word_count > 20:  # At least 20 words following = likely real content
            # Find where this section ends
            end_match = end_pattern.search(text, pos=match.end())
            if end_match:
                section_length = end_match.start() - match.start()
            else:
                section_length = len(text) - match.start()

            if section_length > max_length:
                max_length = section_length
                best_match = match

    if not best_match:
        logger.warning("Found Item 1A markers but couldn't identify main section. Using first match.")
        best_match = matches[0]

    # Find the end marker ONLY after the start marker
    end_match = end_pattern.search(text, pos=best_match.end())

    if end_match:
        logger.info(f"Successfully sliced Item 1A section ({max_length} characters)")
        return text[best_match.start() : end_match.start()].strip()

    logger.info("Item 1A found but no end marker detected. Slicing from 1A to EOF.")
    return text[best_match.start() :].strip()

def validate_risk_factors_text(text: str, config: EdgarConfig) -> bool:
    """
    Validate extracted risk factors text meets quality criteria.
    
    Checks:
    - Word count within acceptable range
    - Minimum sentence count
    - Contains "risk" keyword if required
    - Not a TOC entry pattern
    
    Args:
        text: Extracted risk factors text
        config: Edgar configuration with validation criteria
        
    Returns:
        True if text passes all validation checks, False otherwise
    """
    if not text:
        return False
    
    # Check for TOC-like patterns (e.g., "Item 1A. Risk Factors...14")
    # TOC entries typically have dots leading to page numbers or very short content
    toc_patterns = [
        r'risk\s+factors[\s\.]+\d+$',  # "Risk Factors...14" or "RISK FACTORS 23"
        r'item\s+1a[\s\.].*page\s+\d+',  # "Item 1A. Risk Factors Page 45"
        r'\.{3,}',  # Multiple dots (leader dots in TOC)
    ]
    text_lower = text.lower()
    for pattern in toc_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            logger.debug(f"Rejected: TOC pattern detected ({pattern})")
            return False
    
    # Word count validation
    words = text.split()
    word_count = len(words)
    
    if word_count < config.validation.min_words:
        logger.debug(f"Rejected: {word_count} words < minimum {config.validation.min_words}")
        return False
    
    if word_count > config.validation.max_words:
        logger.debug(f"Rejected: {word_count} words > maximum {config.validation.max_words}")
        return False
    
    # Sentence count validation (simple heuristic: count periods, exclamation, question marks)
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    
    if sentence_count < config.validation.min_sentences:
        logger.debug(f"Rejected: {sentence_count} sentences < minimum {config.validation.min_sentences}")
        return False
    
    # Keyword validation
    if config.validation.must_contain_risk:
        if 'risk' not in text_lower:
            logger.debug("Rejected: does not contain 'risk' keyword")
            return False
    
    return True

def extract_risk_factors_via_edgartools(
    ticker: str, 
    year: int, 
    config: EdgarConfig
) -> Optional[str]:
    """
    Extract Item 1A (Risk Factors) using edgartools library's DOM parser.
    
    This provides superior accuracy over regex-based extraction for many filings
    by using the document's native structure/outline.
    
    Args:
        ticker: Company ticker symbol (e.g., "AAPL")
        year: Filing year
        config: Edgar configuration with identity and validation settings
        
    Returns:
        Extracted risk factors text if successful and valid, None otherwise
    """
    try:
        # Import edgartools dynamically (optional dependency)
        from edgar import Company, set_identity
    except ImportError:
        logger.info("edgartools not installed, skipping DOM-based extraction")
        return None
    
    try:
        # Set SEC-required identity for API compliance
        set_identity(config.identity_email)
        
        # Fetch company and get latest 10-K filing for the year
        company = Company(ticker)
        filings = company.get_filings(form="10-K")
        filing = filings.latest(year)
        
        if not filing:
            logger.warning(f"No 10-K filing found for {ticker} in {year}")
            return None
        
        # Extract 10-K object and get risk_factors property
        tenk = filing.obj()
        risk_text = tenk.risk_factors
        
        if not risk_text:
            logger.warning(f"edgartools: No risk_factors property for {ticker} {year}")
            return None
        
        # Validate the extracted content
        if not validate_risk_factors_text(risk_text, config):
            logger.warning(f"edgartools: Extracted text failed validation for {ticker} {year}")
            return None
        
        logger.info(f"edgartools: Successfully extracted {len(risk_text)} chars for {ticker} {year}")
        return risk_text
        
    except Exception as e:
        logger.warning(f"edgartools extraction failed for {ticker} {year}: {e}")
        return None

def extract_risk_factors_with_fallback(
    ticker: str,
    year: int,
    html_path: str,
    config: Config
) -> Tuple[str, str]:
    """
    Orchestrator: Extract Item 1A using edgartools first, fallback to regex-based extraction.
    
    Strategy:
    1. If edgartools enabled and available, try DOM-based extraction
    2. If that succeeds and validates, return result with method="edgartools"
    3. Otherwise, fallback to existing slice_risk_factors() logic with method="fallback"
    
    Args:
        ticker: Company ticker symbol
        year: Filing year
        html_path: Path to downloaded HTML file (for fallback)
        config: Full application config (includes edgar settings)
        
    Returns:
        Tuple of (extracted_text, extraction_method)
        where method is either "edgartools" or "fallback"
    """
    # Try edgartools if enabled
    if config.edgar.enabled:
        logger.info(f"Attempting edgartools extraction for {ticker} {year}")
        edgartools_result = extract_risk_factors_via_edgartools(ticker, year, config.edgar)
        
        if edgartools_result:
            logger.info(f"Using edgartools extraction for {ticker} {year}")
            return (edgartools_result, "edgartools")
        
        logger.info(f"edgartools extraction failed or unavailable, falling back to regex for {ticker} {year}")
    else:
        logger.info(f"edgartools disabled, using regex extraction for {ticker} {year}")
    
    # Fallback to regex-based extraction
    full_text = extract_text_from_file(html_path)
    risk_text = slice_risk_factors(full_text)
    
    logger.info(f"Using fallback extraction for {ticker} {year} ({len(risk_text)} chars)")
    return (risk_text, "fallback")

def run_ingestion_pipeline(html_content: str, ticker: str, year: int) -> str:
    """
    Orchestrates the flow from raw HTML to Vector DB atoms.
    """
    # 1. Extraction: From raw HTML to clean 'Item 1A' text
    # We modularize the calls to your existing logic
    full_text = parse_sec_html(html_content)
    risk_text = slice_risk_factors(full_text)

    if not risk_text or len(risk_text) < 100:
        logger.warning(f"Risk section for {ticker} seems suspiciously short or missing.")
        # Return or handle error as per your preference

    # 2. Chunking: From long text to metadata-tagged atoms (Subissue 1.0)
    metadata = {
        "ticker": ticker,
        "year": year,
        "item_type": "1A"
    }
    chunks = chunk_risk_section(risk_text, metadata)

    # 3. Storage & Embeddings (Subissue 1.1 & 1.2)
    # Placeholder for the upcoming 'Vault' logic
    # upsert_to_vault(chunks)

    logger.info(f"Pipeline: Processed {len(chunks)} chunks for {ticker} {year}")
    return f"Processed {len(chunks)} chunks for {ticker}"
