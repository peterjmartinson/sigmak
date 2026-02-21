#!/usr/bin/env python3
"""
Generate Investment-Grade Risk Analysis Report for SEC 10-K Filings.

This script analyzes SEC 10-K Risk Factor disclosures and produces an
investment-focused report for hedge fund analysts and portfolio managers:
- Material risk assessment for the latest filing
- Risk category concentration analysis
- Identification of emerging vs. resolved risks
- Business impact implications

Usage:
    python generate_yoy_report.py

For custom companies:
    python generate_yoy_report.py <ticker> <year1> <year2> <year3>

Example:
    python generate_yoy_report.py HURC 2023 2024 2025
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, TYPE_CHECKING
import re
import json
from collections import defaultdict
from dotenv import load_dotenv
from sigmak import filings_db
from sigmak.filings_db import get_peer
from sigmak.integration import IntegrationPipeline
from sigmak.ingest import extract_risk_factors_with_fallback
from sigmak.risk_classification_service import RiskClassificationService
import os
from sigmak.integration import RiskAnalysisResult

if TYPE_CHECKING:
    from sigmak.integration import RiskAnalysisResult

# Load environment variables from .env file
load_dotenv()

# Module-level logger
logger = logging.getLogger(__name__)


def validate_cached_result(data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate that cached JSON has required fields.
    
    Args:
        data: Loaded JSON data
        
    Returns:
        Tuple of (is_valid, reason_if_invalid)
    """
    risks = data.get('risks', [])
    if not risks:
        return True, ""  # Empty results are valid
    
    # Check for classification fields
    missing_classification = False
    missing_llm_fields = False
    
    for risk in risks:
        # Check if risk has category
        if not risk.get('category'):
            missing_classification = True
        
        # Check if risk has LLM evidence/rationale when classified by LLM
        if risk.get('classification_method') == 'llm':
            if not risk.get('llm_evidence') or not risk.get('llm_rationale'):
                missing_llm_fields = True
                break
    
    if missing_classification:
        return False, "missing risk classification"
    if missing_llm_fields:
        return False, "missing llm_evidence or llm_rationale fields"
    
    return True, ""


def enrich_result_with_classification(
    result: "RiskAnalysisResult",
    pipeline: IntegrationPipeline,
    force_llm: bool = False
) -> "RiskAnalysisResult":
    """
    Enrich result with classification, preserving existing LLM fields.
    
    Args:
        result: Risk analysis result to enrich
        pipeline: Integration pipeline instance
        force_llm: Force LLM classification even if cached
        
    Returns:
        Enriched result with classifications and LLM evidence/rationale
    """
    use_full_service = False
    classification_service = None
    if os.getenv('GOOGLE_API_KEY'):
        try:
            classification_service = RiskClassificationService(drift_system=pipeline.drift_system)
            use_full_service = True
        except Exception:
            use_full_service = False

    for r in result.risks:
        try:
            # Skip if already has valid classification and LLM fields
            if not force_llm and r.get('category') and r.get('category') != 'UNCATEGORIZED':
                if r.get('classification_method') != 'llm':
                    # Non-LLM classification, keep as-is
                    continue
                if r.get('llm_evidence') and r.get('llm_rationale'):
                    # LLM classification with evidence/rationale, keep as-is
                    continue
            
            # Need to classify or re-classify
            if r.get('metadata', {}).get('category'):
                r['category'] = r['metadata']['category']
                r['category_confidence'] = r['metadata'].get('confidence', 0.0)
                r['classification_method'] = r['metadata'].get('classification_source', 'vector')
                continue

            if use_full_service and classification_service:
                try:
                    llm_result, source = classification_service.classify_with_cache_first(r['text'])
                    r['category'] = llm_result.category.value
                    r['category_confidence'] = llm_result.confidence
                    r['classification_method'] = source
                    
                    # CRITICAL: Always capture LLM evidence and rationale
                    if llm_result.evidence:
                        r['llm_evidence'] = llm_result.evidence
                    if llm_result.rationale:
                        r['llm_rationale'] = llm_result.rationale
                        
                except Exception as e:
                    print(f"   ⚠️  Classification failed for risk: {e}")
                    r['category'] = 'UNCATEGORIZED'
                    r['category_confidence'] = 0.0
                    r['classification_method'] = 'error'
            else:
                # Cache-only classification via DriftDetectionSystem
                try:
                    embedding = pipeline.indexing_pipeline.embeddings.encode([r['text']])[0].tolist()
                    cache_results = pipeline.drift_system.similarity_search(query_embedding=embedding, n_results=1)
                    if cache_results and cache_results[0].get('similarity_score', 0.0) >= 0.8:
                        top = cache_results[0]
                        r['category'] = top.get('category', 'UNCATEGORIZED')
                        r['category_confidence'] = float(top.get('confidence', 0.0))
                        r['classification_method'] = 'vector_db'
                        # Try to get LLM fields from cache if available
                        if top.get('evidence'):
                            r['llm_evidence'] = top['evidence']
                        if top.get('rationale'):
                            r['llm_rationale'] = top['rationale']
                    else:
                        r['category'] = 'UNCATEGORIZED'
                        r['category_confidence'] = 0.0
                        r['classification_method'] = 'db_only_no_match'
                except Exception as e:
                    print(f"   ⚠️  Cache lookup failed for risk: {e}")
                    r['category'] = 'UNCATEGORIZED'
                    r['category_confidence'] = 0.0
                    r['classification_method'] = 'error'
        except Exception as e:
            print(f"   ⚠️  Error enriching risk: {e}")
            continue

    return result


def load_or_analyze_filing(
    pipeline: IntegrationPipeline,
    html_path: str,
    ticker: str,
    year: int,
    retrieve_top_k: int = 10,
    use_llm: bool = False
) -> "RiskAnalysisResult":
    """
    Load cached results or analyze filing fresh.
    
    Args:
        pipeline: Integration pipeline instance
        html_path: Path to HTML filing
        ticker: Stock ticker symbol
        year: Filing year
        retrieve_top_k: Number of top risks to retrieve
        use_llm: Whether LLM classification is enabled (bypasses cache)
        
    Returns:
        RiskAnalysisResult with full risk analysis
    """
    # Note: `IntegrationPipeline` is imported at module level; keep imports minimal here.

    cache_file = Path(f"output/results_{ticker}_{year}.json")
    
    # Skip cache when LLM is enabled to ensure fresh enrichment
    if not use_llm and cache_file.exists():
        print(f"📂 Loading cached results for {ticker} {year}...")
        with open(cache_file, 'r') as f:
            data = json.load(f)
            
            # Validate cached results
            is_valid, reason = validate_cached_result(data)
            
            # If cached result was generated without reranking enabled,
            # re-run analysis to ensure reranking is applied (default: rerank=True).
            if not data.get('metadata', {}).get('rerank', True):
                print(f"♻️ Cached results for {ticker} {year} were generated with rerank=False; re-running with rerank=True...")
                result = pipeline.analyze_filing(
                    html_path=html_path,
                    ticker=ticker,
                    filing_year=year,
                    retrieve_top_k=retrieve_top_k,
                    rerank=True
                )
                # Enrich with classification
                result = enrich_result_with_classification(result, pipeline)
                # Update cache
                with open(cache_file, 'w') as wf:
                    wf.write(result.to_json())
                return result
            
            # If validation failed, re-enrich and update cache
            if not is_valid:
                print(f"⚠️  Cached results for {ticker} {year} incomplete ({reason}); re-enriching...")
                result = RiskAnalysisResult(
                    ticker=data['ticker'],
                    filing_year=data['filing_year'],
                    risks=data['risks'],
                    metadata=data['metadata']
                )
                # Re-enrich with classification to get missing fields
                result = enrich_result_with_classification(result, pipeline, force_llm=True)
                # Update cache with enriched data
                with open(cache_file, 'w') as wf:
                    wf.write(result.to_json())
                print(f"   ✅ Re-enriched and updated cache for {ticker} {year}")
                return result

            return RiskAnalysisResult(
                ticker=data['ticker'],
                filing_year=data['filing_year'],
                risks=data['risks'],
                metadata=data['metadata']
            )
    
    # Analyze fresh
    print(f"🔍 Analyzing {ticker} {year} from {html_path}...")
    result = pipeline.analyze_filing(
        html_path=html_path,
        ticker=ticker,
        filing_year=year,
        retrieve_top_k=retrieve_top_k,
        rerank=True
    )
    
    print(f"   ✅ Indexed {result.metadata['chunks_indexed']} chunks in {result.metadata['total_latency_ms']:.0f}ms")
    
    # Enrich result with classification (preserves LLM evidence/rationale)
    result = enrich_result_with_classification(result, pipeline)
    
    # Cache results with all enrichments
    with open(cache_file, 'w') as f:
        f.write(result.to_json())

    return result


def calculate_risk_similarity(risk1_text: str, risk2_text: str) -> float:
    """
    Calculate simple word overlap similarity between two risk texts.
    
    Args:
        risk1_text: First risk text
        risk2_text: Second risk text
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    # Simple word-based similarity (good enough for year-over-year comparison)
    words1 = set(risk1_text.lower().split())
    words2 = set(risk2_text.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def identify_risk_changes(
    results: List["RiskAnalysisResult"],
    similarity_threshold: float = 0.4
) -> Dict[str, Any]:
    """
    Identify new, disappeared, and persistent risks across years.
    
    Args:
        results: List of RiskAnalysisResult ordered by year
        similarity_threshold: Minimum similarity to consider risks "the same"
        
    Returns:
        Dictionary with new_risks, disappeared_risks, and persistent_risks
    """
    changes = {
        'new_risks': [],
        'disappeared_risks': [],
        'persistent_risks': []
    }
    
    if len(results) < 2:
        return changes
    
    # Compare consecutive years
    for i in range(len(results) - 1):
        prev_result = results[i]
        curr_result = results[i + 1]
        
        prev_risks = prev_result.risks
        curr_risks = curr_result.risks
        
        # Find new risks (in current year but not in previous)
        for curr_risk in curr_risks:
            is_new = True
            for prev_risk in prev_risks:
                similarity = calculate_risk_similarity(
                    curr_risk['text'],
                    prev_risk['text']
                )
                if similarity >= similarity_threshold:
                    is_new = False
                    # Track as persistent if it's evolved
                    changes['persistent_risks'].append({
                        'year': curr_result.filing_year,
                        'prev_year': prev_result.filing_year,
                        'risk': curr_risk,
                        'similarity': similarity
                    })
                    break
            
            if is_new:
                changes['new_risks'].append({
                    'year': curr_result.filing_year,
                    'risk': curr_risk
                })
        
        # Find disappeared risks (in previous year but not in current)
        for prev_risk in prev_risks:
            is_disappeared = True
            for curr_risk in curr_risks:
                similarity = calculate_risk_similarity(
                    prev_risk['text'],
                    curr_risk['text']
                )
                if similarity >= similarity_threshold:
                    is_disappeared = False
                    break
            
            if is_disappeared:
                changes['disappeared_risks'].append({
                    'year': prev_result.filing_year,
                    'risk': prev_risk
                })
    
    return changes


def calculate_category_distribution(result: "RiskAnalysisResult") -> Dict[str, List[Dict[str, Any]]]:
    """
    Calculate distribution of risk categories with associated risks.
    
    Args:
        result: Risk analysis result
        
    Returns:
        Dictionary mapping category names to list of risks in that category
    """
    distribution = defaultdict(list)
    
    for risk in result.risks:
        # Extract category from risk metadata if available
        category = risk.get('category', 'UNCATEGORIZED')
        distribution[category].append(risk)
    
    return dict(distribution)


def extract_category_from_text(risk_text: str) -> str:
    """
    Extract category hint from risk text (fallback if category not in metadata).
    
    Args:
        risk_text: Risk disclosure text
        
    Returns:
        Best guess category name
    """
    text_lower = risk_text.lower()
    
    # Simple keyword matching for category inference
    category_keywords = {
        'OPERATIONAL': ['supply chain', 'manufacturing', 'operations', 'production', 'facilities'],
        'FINANCIAL': ['debt', 'liquidity', 'credit', 'capital', 'financial'],
        'REGULATORY': ['regulation', 'compliance', 'law', 'government', 'legal'],
        'COMPETITIVE': ['competition', 'competitor', 'market share', 'pricing pressure'],
        'TECHNOLOGICAL': ['technology', 'cybersecurity', 'innovation', 'digital', 'it systems'],
        'GEOPOLITICAL': ['international', 'trade', 'tariff', 'sanctions', 'war', 'political'],
        'HUMAN_CAPITAL': ['employee', 'talent', 'workforce', 'labor', 'personnel'],
        'REPUTATIONAL': ['reputation', 'brand', 'customer', 'trust', 'public perception'],
        'SYSTEMATIC': ['economic', 'recession', 'inflation', 'market volatility', 'macroeconomic']
    }
    
    for category, keywords in category_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            return category
    
    return 'OTHER'


def suggest_categories_from_keywords(risk_text: str) -> Tuple[List[str], List[str]]:
    """
    Suggest categories based on keyword matches in risk text.
    
    Args:
        risk_text: Risk disclosure text
        
    Returns:
        Tuple of (detected_keywords, suggested_categories)
    """
    text_lower = risk_text.lower()
    
    category_keywords = {
        'GEOPOLITICAL': ['tariff', 'trade', 'international', 'export', 'import', 'sanctions', 'war', 'political'],
        'OPERATIONAL': ['supply chain', 'supplier', 'manufacturing', 'production', 'facilities', 'operations'],
        'FINANCIAL': ['debt', 'liquidity', 'credit', 'capital', 'cash flow', 'interest'],
        'REGULATORY': ['regulation', 'compliance', 'law', 'government', 'sec', 'regulatory'],
        'COMPETITIVE': ['competition', 'competitor', 'market share', 'pricing', 'rival'],
        'TECHNOLOGICAL': ['technology', 'cyber', 'digital', 'it', 'systems', 'innovation'],
        'HUMAN_CAPITAL': ['employee', 'talent', 'workforce', 'labor', 'personnel', 'hiring'],
        'REPUTATIONAL': ['reputation', 'brand', 'customer', 'trust', 'image'],
        'SYSTEMATIC': ['economic', 'recession', 'inflation', 'market', 'macro']
    }
    
    detected_keywords = []
    category_scores = {}
    
    for category, keywords in category_keywords.items():
        matches = [kw for kw in keywords if kw in text_lower]
        if matches:
            category_scores[category] = len(matches)
            detected_keywords.extend(matches[:3])  # Limit to first 3 per category
    
    # Get top 2 suggested categories
    suggested = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    suggested_categories = [cat for cat, _ in suggested]
    
    return detected_keywords[:3], suggested_categories  # Return max 3 keywords


def load_filing_provenance(ticker: str, filing_year: int, filings_db_path: str | None = None) -> Dict[str, str]:
    """
    Load filing provenance metadata from JSON file if available.
    
    Args:
        ticker: Stock ticker symbol
        filing_year: Filing year
        
    Returns:
        Dictionary with accession, cik, and sec_url (or placeholders)
    """
    # First, attempt to load identifiers from the filings_index SQLite DB
    db_result = filings_db.get_identifiers(filings_db_path, ticker, filing_year)

    # If DB provided values (not missing token), return them
    if db_result and all(v != filings_db.MISSING_TOKEN for v in db_result.values()):
        return db_result

    # Otherwise, try the legacy JSON search as a fallback
    search_dirs = [Path("data/filings"), Path("data"), Path("data/samples")]

    # Common filename patterns
    patterns = [
        f"{ticker.lower()}-{filing_year}*.json",
        f"{ticker.upper()}-{filing_year}*.json",
    ]

    metadata = {
        'accession': db_result.get('accession', filings_db.MISSING_TOKEN),
        'cik': db_result.get('cik', filings_db.MISSING_TOKEN),
        'sec_url': db_result.get('sec_url', filings_db.MISSING_TOKEN)
    }

    # Look for JSON files matching the patterns in all search directories
    for data_dir in search_dirs:
        if not data_dir.exists():
            continue
        for pattern in patterns:
            matches = list(data_dir.glob(pattern))
            if matches:
                try:
                    with open(matches[0], 'r') as f:
                        data = json.load(f)
                        metadata['accession'] = data.get('accession', metadata['accession'])
                        metadata['cik'] = data.get('cik', metadata['cik'])
                        metadata['sec_url'] = data.get('sec_url', metadata['sec_url'])
                        return metadata  # Return immediately on first match
                except (json.JSONDecodeError, IOError):
                    pass

    return metadata


def is_valid_risk_paragraph(text: str) -> bool:
    """
    Filter out boilerplate intro and TOC-like text from Item 1A.
    
    Returns False for:
    - Short paragraphs (< 50 words) likely to be headers/TOC
    - Item 1A title patterns
    - TOC patterns (dots leading to numbers, page references)
    - Generic intro statements without substance
    
    Args:
        text: Risk paragraph text to validate
        
    Returns:
        True if text appears to be a substantive risk disclosure
    """
    text_lower = text.lower().strip()
    
    # Minimum length heuristic (< 50 words = likely header/TOC)
    if len(text.split()) < 50:
        return False
    
    # Exclude Item 1A title patterns
    if re.match(r'^item\s+1a[.\s\-:]+risk\s+factors', text_lower):
        return False
    
    # Exclude TOC patterns (dots leading to numbers, page references)
    if re.search(r'\.{3,}|\bpage\s+\d+', text_lower):
        return False
    
    # Exclude pure list-of-contents statements (if short)
    if re.search(r'^(this section|the following|we face|risks include)', text_lower):
        if len(text.split()) < 80:  # Allow if substantive
            return False
    
    return True


def generate_markdown_report(
    ticker: str,
    results: List["RiskAnalysisResult"],
    output_file: str = "risk_analysis_report.md",
    filings_db_path: str | None = None,
) -> None:
    """
    Generate investment-grade markdown report focusing on latest year's risks.
    
    Args:
        ticker: Stock ticker symbol
        results: List of RiskAnalysisResult ordered by year (oldest to newest)
        output_file: Output markdown file path
    """
    results_sorted = sorted(results, key=lambda r: r.filing_year)
    years = [r.filing_year for r in results_sorted]
    latest_result = results_sorted[-1]
    latest_year = latest_result.filing_year
    changes = identify_risk_changes(results_sorted)
    
    # Build the report
    report_lines = []
    
    # Fetch company name from peers table
    peer_info = get_peer(filings_db_path, ticker) if filings_db_path else None
    company_name = peer_info.get('name', ticker) if peer_info else ticker
    
    # Header
    report_lines.append(f"# {company_name} ({ticker}) — Risk Factor Analysis")
    report_lines.append(f"## Item 1A Deep Dive: {latest_year} 10-K")
    report_lines.append("")
    report_lines.append(f"**Report Date:** {datetime.now().strftime('%B %d, %Y')}")
    
    # Add provenance metadata
    provenance = load_filing_provenance(ticker, latest_year, filings_db_path)
    sec_url = provenance['sec_url']
    if not sec_url.startswith('http') and sec_url != '<SEC_URL>':
        sec_url = f"https://www.sec.gov{sec_url if sec_url.startswith('/') else '/' + sec_url}"
    
    report_lines.append(f"**Filing:** 10-K (Accession: {provenance['accession']}) • CIK: {provenance['cik']} • SEC URL: {sec_url}")
    # Warn in report if any provenance identifiers are missing
    missing_token = filings_db.MISSING_TOKEN
    if any(provenance.get(k) == missing_token for k in ('accession', 'cik', 'sec_url')):
        # Console warning to alert the user during run
        print(f"⚠️  Missing provenance identifiers for {ticker} {latest_year} — see output/missing_identifiers.csv")
        sys.stdout.flush()
        report_lines.append("")
        report_lines.append("**⚠️  Missing provenance identifiers:** One or more filing identifiers (accession, CIK, or SEC URL) were not found in the local filings DB.")
        report_lines.append("See `output/missing_identifiers.csv` for audit details and run the downloader to populate missing records.")
        report_lines.append("")
    report_lines.append(f"**Historical Comparison:** {years[0]}–{years[-1]}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Executive Summary
    report_lines.append("## Executive Summary")
    report_lines.append("")
    
    # Calculate stats for latest year
    latest_severities = [r['severity']['value'] for r in latest_result.risks if 'severity' in r]
    latest_novelties = [r['novelty']['value'] for r in latest_result.risks if 'novelty' in r]
    avg_severity = sum(latest_severities) / len(latest_severities) if latest_severities else 0
    avg_novelty = sum(latest_novelties) / len(latest_novelties) if latest_novelties else 0
    
    total_new = len([r for r in changes['new_risks'] if r['year'] == latest_year])
    total_disappeared = len(changes['disappeared_risks'])
    
    # High/medium/low severity buckets
    high_severity = len([s for s in latest_severities if s >= 0.7])
    medium_severity = len([s for s in latest_severities if 0.4 <= s < 0.7])
    low_severity = len([s for s in latest_severities if s < 0.4])
    
    report_lines.append("### Key Takeaways")
    report_lines.append("")
    
    # Generate PM takeaway based on top risks and changes
    category_dist = calculate_category_distribution(latest_result)
    sorted_cats = sorted(
        category_dist.items(),
        key=lambda x: len(x[1]) * (sum(r.get('severity', {}).get('value', 0) for r in x[1]) / len(x[1]) if x[1] else 0),
        reverse=True
    )
    
    top_category = sorted_cats[0][0] if sorted_cats else "OPERATIONAL"
    top_cat_count = len(sorted_cats[0][1]) if sorted_cats else 0
    
    # Build concise takeaway
    if total_new > 0 and high_severity >= 3:
        what = f"{total_new} new {top_category.lower()} risk(s)"
        why = f"{high_severity} material disclosures"
        action = f"track {top_category.lower()} metrics"
    elif high_severity >= len(latest_result.risks) * 0.3:
        what = f"{high_severity} material {top_category.lower()} risks"
        why = f"{top_cat_count} disclosures in category"
        action = f"monitor {top_category.lower()} exposure"
    else:
        what = f"{top_cat_count} {top_category.lower()} disclosures"
        why = f"primary risk concentration"
        action = f"watch {top_category.lower()} trends"
    
    takeaway = f"**TAKEAWAY:** {what}; {why}; {action}."
    if len(takeaway) > 150:  # Fallback if too long
        takeaway = f"**TAKEAWAY:** {high_severity} material risks in {top_category}; monitor closely."
    
    report_lines.append(takeaway)
    report_lines.append("")
    
    # Alert-style summary bullets
    report_lines.append(f"**FILING SNAPSHOT:** {len(latest_result.risks)} total risk factors | {high_severity} material (≥0.70 severity) | Avg severity: {avg_severity:.2f}")
    report_lines.append("")
    
    if total_new > 0:
        report_lines.append(f"**[ALERT] NEW:** {total_new} risk factor(s) added — signals shifting exposure; review disclosure language for operational changes.")
    
    if total_disappeared > 0:
        report_lines.append(f"**[RESOLVED]:** {total_disappeared} risk factor(s) dropped from disclosure — potential business improvement or strategic pivot.")
    
    # Safety check: only show alert if we have risks to analyze
    if len(latest_result.risks) > 0 and high_severity >= len(latest_result.risks) * 0.4:
        report_lines.append(f"**[ALERT]:** {100*high_severity/len(latest_result.risks):.0f}% of disclosures are high severity — elevated risk profile requires close portfolio monitoring.")
    
    report_lines.append("")
    report_lines.append("### Risk Severity Distribution")
    report_lines.append("")
    report_lines.append("| Severity Level | Count | % of Total |")
    report_lines.append("|----------------|-------|------------|")
    total_risks = len(latest_severities)
    
    # Safety check: prevent division by zero
    if total_risks > 0:
        report_lines.append(f"| High (≥0.70) | {high_severity} | {100*high_severity/total_risks:.0f}% |")
        report_lines.append(f"| Medium (0.40-0.69) | {medium_severity} | {100*medium_severity/total_risks:.0f}% |")
        report_lines.append(f"| Low (<0.40) | {low_severity} | {100*low_severity/total_risks:.0f}% |")
    else:
        report_lines.append("| High (≥0.70) | 0 | 0% |")
        report_lines.append("| Medium (0.40-0.69) | 0 | 0% |")
        report_lines.append("| Low (<0.40) | 0 | 0% |")
    report_lines.append("")
    
    # Add severity legend and percentile
    report_lines.append("**Severity Legend:** [CRIT] ≥0.80 | [HIGH] 0.70–0.79 | [MED] 0.50–0.69 | [LOW] <0.50")
    report_lines.append("")
    report_lines.append(f"**{ticker} avg severity percentile vs. universe:** <placeholder percentile>")
    report_lines.append("")
    
    # Category distribution
    category_dist = calculate_category_distribution(latest_result)
    
    report_lines.append("### Risk Concentration by Category")
    report_lines.append("")
    
    # Sort categories by weighted severity (count × avg severity)
    sorted_categories = sorted(
        category_dist.items(),
        key=lambda x: len(x[1]) * (sum(r.get('severity', {}).get('value', 0) for r in x[1]) / len(x[1]) if x[1] else 0),
        reverse=True
    )
    
    # Find most concentrated category
    if sorted_categories:
        top_category = sorted_categories[0][0]
        top_count = len(sorted_categories[0][1])
        report_lines.append(f"**Primary Risk Exposure**: {top_category} ({top_count} disclosures)")
        report_lines.append("")
    
    report_lines.append("| Category | Risk Count | Avg Severity | Material Risks (≥0.70) |")
    report_lines.append("|----------|------------|--------------|------------------------|")
    
    for category, risks in sorted_categories:
        count = len(risks)
        severities = [r.get('severity', {}).get('value', 0) for r in risks]
        avg_cat_severity = sum(severities) / len(severities) if severities else 0
        material_count = len([s for s in severities if s >= 0.7])
        
        # Handle uncategorized with suggestions
        display_category = category
        if category in ['UNCATEGORIZED', 'OTHER']:
            # Get keyword suggestions from first risk in category
            first_risk = risks[0] if risks else None
            if first_risk:
                keywords, suggestions = suggest_categories_from_keywords(first_risk['text'])
                if suggestions:
                    display_category = f"Uncategorized (low classifier confidence) — suggested: {', '.join(suggestions)}"
                else:
                    display_category = "Uncategorized (low classifier confidence)"
        
        report_lines.append(f"| {display_category} | {count} | {avg_cat_severity:.2f} | {material_count} |")
    
    # Add keyword mapping note for uncategorized if present
    has_uncategorized = any(cat in ['UNCATEGORIZED', 'OTHER'] for cat, _ in sorted_categories)
    if has_uncategorized:
        for category, risks in sorted_categories:
            if category in ['UNCATEGORIZED', 'OTHER'] and risks:
                keywords, suggestions = suggest_categories_from_keywords(risks[0]['text'])
                if keywords:
                    kw_str = ', '.join(keywords)
                    sugg_str = '/'.join(suggestions) if suggestions else 'N/A'
                    report_lines.append("")
                    report_lines.append(f"_Keywords detected: {kw_str} → suggest {sugg_str}_")
                break
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Material Risk Factors (Latest Year) - Sorted by Novelty
    report_lines.append(f"## Material Risk Factors — {latest_year}")
    report_lines.append("")
    
    # Load provenance for evidence links
    provenance = load_filing_provenance(ticker, latest_year, filings_db_path)
    base_sec_url = provenance['sec_url']
    if base_sec_url == '<SEC_URL>':
        base_sec_url = f"https://www.sec.gov/Archives/edgar/data/{provenance['cik']}/{provenance['accession'].replace('-', '')}/{provenance['accession']}.htm"
    
    # Filter out boilerplate/intro paragraphs before sorting
    valid_risks = [r for r in latest_result.risks if is_valid_risk_paragraph(r['text'])]
    
    # Filter out BOILERPLATE category (TOC, headers, metadata)
    boilerplate_risks = [r for r in valid_risks if r.get('category', '').lower() == 'boilerplate']
    valid_risks = [r for r in valid_risks if r.get('category', '').lower() != 'boilerplate']
    
    # Log boilerplate filtering for visibility
    if boilerplate_risks:
        logging.info(f"Filtered {len(boilerplate_risks)} BOILERPLATE chunks from {ticker} {latest_year}")
        for bp in boilerplate_risks:
            logging.debug(f"  Boilerplate: {bp['text'][:80]}...")
    
    # Sort by novelty (most novel first) instead of severity
    sorted_risks = sorted(
        valid_risks,
        key=lambda r: r.get('novelty', {}).get('value', 0),
        reverse=True
    )
    
    # Handle zero-risk case
    num_risks = len(sorted_risks)
    if num_risks == 0:
        report_lines.append("**No material risks detected after filtering boilerplate.**")
        report_lines.append("")
    else:
        report_lines.append(f"The following {num_risks} risk factor(s) are ordered by novelty (most novel/unprecedented first):")
        report_lines.append("")
    
    for i, risk in enumerate(sorted_risks, 1):
        severity = risk.get('severity', {})
        novelty = risk.get('novelty', {})
        category = risk.get('category', extract_category_from_text(risk['text']))
        text_preview = risk['text'][:250].replace('\n', ' ')
        
        severity_val = severity.get('value', 0)
        novelty_val = novelty.get('value', 0)
        
        # Determine risk label (ASCII-only for PDF compatibility)
        if severity_val >= 0.8:
            risk_label = "[CRIT]"
        elif severity_val >= 0.7:
            risk_label = "[HIGH]"
        elif severity_val >= 0.5:
            risk_label = "[MED]"
        else:
            risk_label = "[LOW]"
        
        # Determine status badge (ASCII-only for PDF compatibility)
        status_badge = ""
        if novelty_val >= 0.50:
            status_badge = " [NEW]"
        
        # Check if this risk was in previous years (dropped status)
        for dropped in changes['disappeared_risks']:
            dropped_text = dropped['risk']['text']
            similarity = calculate_risk_similarity(risk['text'], dropped_text)
            if similarity >= 0.4:
                dropped_year = dropped['year']
                status_badge = f" — DROPPED (previously disclosed in {dropped_year})"
                break
        
        # Determine confidence level
        if severity_val >= 0.80 and novelty_val >= 0.20:
            confidence = "High"
        elif (0.50 <= severity_val < 0.80) or (0.10 <= novelty_val <= 0.49):
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Count supporting quotes (default to 1 for the current quote)
        supporting_quotes = risk.get('metadata', {}).get('chunk_count', 1)
        
        # NOTE: evidence_url removed - paragraph anchors were inaccurate (used enumerate index not real para IDs)
        # TODO: Re-enable when paragraph IDs can be accurately mapped to SEC filing anchors
        
        # Generate impact label from severity explanation
        severity_explanation = severity.get('explanation', 'material business impact')
        impact_keywords = {
            'revenue': 'revenue impact',
            'earnings': 'earnings impact',
            'profit': 'margin compression',
            'cash': 'liquidity risk',
            'operational': 'operational disruption',
            'supply chain': 'supply disruption',
            'competition': 'competitive pressure',
            'regulatory': 'compliance risk',
            'market': 'market volatility',
            'customer': 'customer attrition',
            'financial': 'financial stress'
        }
        
        impact_label = "material business impact"
        for keyword, label in impact_keywords.items():
            if keyword in severity_explanation.lower():
                impact_label = label
                break
        
        # Severity-based qualifier
        if severity_val >= 0.8:
            impact_label = f"{impact_label} (High)"
        elif severity_val >= 0.7:
            impact_label = f"{impact_label} (Med-High)"
        else:
            impact_label = f"{impact_label} (Moderate)"
        
        # Category-based monitoring signals
        monitoring_signals = {
            'OPERATIONAL': 'production metrics, supplier performance',
            'FINANCIAL': 'liquidity ratios, credit ratings',
            'REGULATORY': 'regulatory filings, policy announcements',
            'COMPETITIVE': 'market share data, pricing trends',
            'TECHNOLOGICAL': 'cybersecurity incidents, tech investments',
            'GEOPOLITICAL': 'trade policy, tariff announcements',
            'HUMAN_CAPITAL': 'turnover rates, hiring metrics',
            'REPUTATIONAL': 'brand sentiment, customer reviews',
            'SYSTEMATIC': 'macro indicators, interest rates',
            'OTHER': 'company disclosures, industry trends'
        }
        
        monitoring = monitoring_signals.get(category, 'quarterly filings, management guidance')
        
        # Build short evidence link
        evidence_short = f"para {i}"
        
        report_lines.append(f"### {i}. {risk_label} — {category}{status_badge}")
        report_lines.append("")
        report_lines.append(f"_Severity: {severity_val:.2f} | Novelty: {novelty_val:.2f} | Confidence: {confidence}_")
        report_lines.append("")
        report_lines.append(f"> {text_preview}...")
        report_lines.append("")
        
        # Add LLM classification reasoning if available
        llm_evidence = risk.get('llm_evidence', '')
        llm_rationale = risk.get('llm_rationale', '')
        
        if llm_rationale:
            report_lines.append(f"**Classification Rationale:** {llm_rationale}")
            report_lines.append("")
        
        if llm_evidence:
            # Clean and truncate evidence if too long
            evidence_text = llm_evidence.replace('\n', ' ').strip()
            if len(evidence_text) > 400:
                evidence_text = evidence_text[:397] + "..."
            report_lines.append(f"**Key Risk Factors:** {evidence_text}")
            report_lines.append("")
        
        # Filing citation removed - see note above about inaccurate paragraph anchors
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
    
    # Category Deep Dives
    report_lines.append("## Risk Category Breakdown")
    report_lines.append("")
    
    for category, risks in sorted_categories[:5]:  # Top 5 categories only
        if not risks:
            continue
            
        severities = [r.get('severity', {}).get('value', 0) for r in risks]
        avg_severity = sum(severities) / len(severities) if severities else 0
        
        # Handle uncategorized display
        display_category = category
        if category in ['UNCATEGORIZED', 'OTHER']:
            keywords, suggestions = suggest_categories_from_keywords(risks[0]['text'])
            if suggestions:
                display_category = f"Uncategorized (low classifier confidence) — suggested mapping: {', '.join(suggestions)}"
                report_lines.append(f"### {display_category}")
                report_lines.append("")
                report_lines.append(f"**{len(risks)} risks, avg severity: {avg_severity:.2f}**")
                if keywords:
                    kw_str = ', '.join(keywords)
                    sugg_str = '/'.join(suggestions)
                    report_lines.append("")
                    report_lines.append(f"_Keywords detected: {kw_str} → suggest {sugg_str}_")
            else:
                report_lines.append(f"### Uncategorized (low classifier confidence)")
                report_lines.append("")
                report_lines.append(f"**{len(risks)} risks, avg severity: {avg_severity:.2f}**")
        else:
            report_lines.append(f"### {category} ({len(risks)} risks, avg severity: {avg_severity:.2f})")
        
        report_lines.append("")
        
        # Get top 2 risks in this category
        top_risks = sorted(
            risks,
            key=lambda r: r.get('severity', {}).get('value', 0),
            reverse=True
        )[:2]
        
        for risk in top_risks:
            severity = risk.get('severity', {}).get('value', 0)
            text_preview = risk['text'][:180].replace('\n', ' ')
            
            report_lines.append(f"- **[{severity:.2f}]** {text_preview}...")
        
        report_lines.append("")
    
    # Disappeared Risks Section (from historical context)
    if changes['disappeared_risks']:
        report_lines.append("## Resolved or De-emphasized Risks")
        report_lines.append("")
        report_lines.append(f"The following risks were disclosed in prior filings but removed in {latest_year}, potentially signaling improved business conditions or strategic pivots:")
        report_lines.append("")
        
        for disappeared_risk in changes['disappeared_risks'][:8]:  # Limit to top 8
            year = disappeared_risk['year']
            risk = disappeared_risk['risk']
            category = risk.get('category', extract_category_from_text(risk['text']))
            text_preview = risk['text'][:180].replace('\n', ' ')
            
            report_lines.append(f"**[{category}] Last seen: {year}**")
            report_lines.append(f"> {text_preview}...")
            report_lines.append("")
    
    # Comparison Table
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## Multi-Year Disclosure Trends")
    report_lines.append("")
    report_lines.append("| Year | Risk Factors | High Severity (≥0.70) | Avg Severity |")
    report_lines.append("|------|--------------|----------------------|--------------|")
    
    for result in results_sorted:
        year = result.filing_year
        total = len(result.risks)
        sev_values = [r.get('severity', {}).get('value', 0) for r in result.risks]
        high_sev = len([s for s in sev_values if s >= 0.7])
        avg_sev = sum(sev_values) / len(sev_values) if sev_values else 0
        
        report_lines.append(f"| {year} | {total} | {high_sev} | {avg_sev:.2f} |")
    
    report_lines.append("")
    report_lines.append("")
    
    # Methodology
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## About This Analysis")
    report_lines.append("")
    report_lines.append("This report analyzes Item 1A (Risk Factors) disclosures from SEC 10-K filings using proprietary natural language processing and semantic analysis. Each risk factor is:")
    report_lines.append("")
    report_lines.append("1. **Extracted** from the company's official SEC filing")
    report_lines.append("2. **Scored** for severity (potential business impact) and novelty (uniqueness vs. historical patterns)")
    report_lines.append("3. **Classified** into standard risk categories for cross-company comparison")
    report_lines.append("4. **Compared** against prior year disclosures to identify emerging or resolved risks")
    report_lines.append("")
    report_lines.append("**Severity scores** (0.0–1.0) reflect the potential magnitude of business impact, with ≥0.70 considered material.")
    report_lines.append("")
    report_lines.append("**Novelty scores** (0.0–1.0) indicate how unprecedented or unique a disclosure is relative to the company's historical risk profile and industry norms.")
    report_lines.append("")
    report_lines.append("### Risk Category Framework")
    report_lines.append("")
    report_lines.append("- **OPERATIONAL**: Supply chain, manufacturing, operations")
    report_lines.append("- **FINANCIAL**: Liquidity, debt, capital structure")
    report_lines.append("- **REGULATORY**: Compliance, legal, policy")
    report_lines.append("- **COMPETITIVE**: Market position, pricing, rivalry")
    report_lines.append("- **TECHNOLOGICAL**: Cybersecurity, innovation, disruption")
    report_lines.append("- **GEOPOLITICAL**: Trade, sanctions, international exposure")
    report_lines.append("- **HUMAN_CAPITAL**: Workforce, talent, labor relations")
    report_lines.append("- **REPUTATIONAL**: Brand, trust, public perception")
    report_lines.append("- **SYSTEMATIC**: Macro conditions, market volatility")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## Legal Disclaimer")
    report_lines.append("")
    report_lines.append("**Use at your own risk.** This report is for informational purposes only and does not constitute investment advice, financial advice, trading advice, or any other sort of advice. We cannot guarantee investment returns; you may lose money. Past performance is not indicative of future results. Always conduct your own research and consult a qualified financial advisor before making investment decisions.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append(f"*Analysis powered by SigmaK Risk Intelligence Platform | {datetime.now().strftime('%B %Y')}*")
    
    # Write to file
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n✨ Report generated: {output_path.absolute()}")


def main():
    """Main entry point for report generation."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate YoY Risk Analysis Report for SEC 10-K filings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_yoy_report.py                          # Default: HURC 2023-2025
    python generate_yoy_report.py TSLA 2022 2023 2024     # Custom ticker and years
    python generate_yoy_report.py HURC 2023 2024 2025
                """
    )
    parser.add_argument(
        'ticker',
        nargs='?',
        default='HURC',
        help='Stock ticker symbol (default: HURC)'
    )
    parser.add_argument(
        'years',
        nargs='*',
        type=int,
        help='Filing years (default: 2023 2024 2025)'
    )
    # Default behavior: vector-store-first, then LLM fallback. Use DB-only flag to prevent LLM calls.
    parser.add_argument(
        '--db-only-classification',
        action='store_true',
        help='Use DB-only classification (never call the LLM)'
    )
    parser.add_argument(
        '--db-only-similarity-threshold',
        type=float,
        default=0.8,
        help='Similarity threshold (0-1) for accepting DB-only vector matches (default: 0.8)'
    )
    
    args = parser.parse_args()
    
    # Setup output log directory and file logging
    log_dir = Path("output/log")
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    script_name = Path(__file__).stem if '__file__' in globals() else 'generate_yoy_report'
    log_file = log_dir / f"{script_name}_{ts}.log"

    # Configure root logger to also write to the file
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Avoid adding multiple handlers if main() is invoked multiple times
    if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == str(log_file) for h in root_logger.handlers):
        fh = logging.FileHandler(str(log_file), encoding='utf-8')
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s')
        fh.setFormatter(fmt)
        root_logger.addHandler(fh)

    # Also emit a small header to the log file
    logging.info(f"Logging to {log_file}")

    # Set defaults
    ticker = args.ticker.upper()
    years = args.years if args.years else [2023, 2024, 2025]
    db_only_classification = args.db_only_classification
    db_only_similarity_threshold = args.db_only_similarity_threshold
    
    # Validate years
    if len(years) < 2:
        parser.error("At least 2 years required for YoY comparison")
    
    # Build file paths (robust recursive search)
    filings = []
    search_dirs = [Path("data/filings"), Path("data"), Path("data/samples")]

    ticker_l = ticker.lower()
    ticker_u = ticker.upper()

    for year in years:
        found = False
        year_str = str(year)

        for base in search_dirs:
            if not base.exists():
                continue

            # Prefer a company subfolder if present
            candidates_dirs = []
            company_dir = base / ticker_u
            if company_dir.exists():
                candidates_dirs.append(company_dir)
            candidates_dirs.append(base)

            for search_base in candidates_dirs:
                # Look for files with both ticker and year in the filename (case-insensitive)
                matches = list(search_base.rglob(f"*{ticker_l}*{year_str}*.htm")) + list(search_base.rglob(f"*{ticker_u}*{year_str}*.htm"))

                # If no direct matches, fall back to any .htm files that include the year and ticker anywhere
                if not matches:
                    matches = [p for p in search_base.rglob("*.htm") if year_str in p.name and ticker_l in p.name.lower()]

                # If still no matches, look for a subfolder named after the year (e.g., data/filings/IBM/2025/) and use .htm files there
                if not matches:
                    year_dir = search_base / year_str
                    if year_dir.exists() and year_dir.is_dir():
                        matches = list(year_dir.glob("*.htm"))

                if matches:
                    # Prefer files that explicitly mention 10-K ("10k" or "x10k") or an expected year-end
                    def score_path(p: Path) -> int:
                        name = p.name.lower()
                        score = 0
                        if "10k" in name or "x10k" in name:
                            score += 10
                        if re.search(r"(1231|1031|0930|0630)", name):
                            score += 5
                        if ticker_l in name:
                            score += 2
                        return score

                    best = max(matches, key=score_path)
                    filings.append((str(best), ticker, year))
                    found = True
                    break

            if found:
                break

        if not found:
            print(f"⚠️  Could not find filing for {ticker} {year}")
            print(f"    Searched under: {', '.join(str(p) for p in search_dirs if p.exists())}")
            sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"📊 SigmaK Year-over-Year Risk Analysis")
    print(f"{'='*60}")
    print(f"Company: {ticker}")
    print(f"Years: {', '.join(map(str, years))}")
    print(f"Classification mode: {'DB-only (no LLM)' if db_only_classification else 'vector-first then LLM fallback'}")
    print(f"{'='*60}\n")
    
    # Initialize pipeline
    pipeline = IntegrationPipeline(
        persist_path="./database",
        db_only_classification=db_only_classification,
        db_only_similarity_threshold=db_only_similarity_threshold
    )
    
    # Analyze each filing
    results = []
    for html_path, ticker_sym, year in filings:
        result = load_or_analyze_filing(
            pipeline=pipeline,
            html_path=html_path,
            ticker=ticker_sym,
            year=year,
            retrieve_top_k=10
        )
        results.append(result)

        # Save extracted Item 1A snippet to output/log if available
        try:
            # Attempt to re-run extraction to obtain raw Item 1A section
            extraction_text, extraction_method = extract_risk_factors_with_fallback(
                ticker=ticker_sym,
                year=year,
                html_path=html_path,
                config=pipeline.indexing_pipeline.config
            )

            if extraction_text and len(extraction_text.strip()) > 0:
                snippet_filename = log_dir / f"{ticker_sym}_{year}_item1a_extract_{ts}.txt"
                with open(snippet_filename, 'w', encoding='utf-8') as sf:
                    sf.write(f"# Extraction method: {extraction_method}\n")
                    sf.write(f"# Source: {html_path}\n")
                    sf.write(f"# Timestamp: {datetime.now().isoformat()}\n\n")
                    sf.write(extraction_text)
                logging.info(f"Saved extracted Item 1A snippet: {snippet_filename}")
        except Exception as e:
            logging.exception(f"Failed to save Item 1A snippet for {ticker_sym} {year}: {e}")
    
    # Generate report
    print(f"\n{'='*60}")
    print("📝 Generating Markdown Report...")
    print(f"{'='*60}\n")
    
    output_file = f"output/{ticker}_YoY_Risk_Analysis_{min(years)}_{max(years)}.md"
    # Use the downloader's filings DB (sec_filings.db) for provenance identifiers
    filings_db_path = "./database/sec_filings.db"
    generate_markdown_report(ticker, results, output_file, filings_db_path)
    
    print(f"\n{'='*60}")
    print("✅ Analysis Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
