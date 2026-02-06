"""Integration tests for YOY report generation.

Tests edge cases and requirements from FIX_YOY_REPORT_BUGS.md:
- Zero-risk handling
- Novelty-based ordering
- Company name in header
- No emojis (PDF compatibility)
- Legal disclaimer
- No Filing Reference URLs
"""

import tempfile
from pathlib import Path
import pytest
import sys

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from generate_yoy_report import generate_markdown_report


# Mock objects for testing without full pipeline
class MockRiskAnalysisResult:
    """Mock RiskAnalysisResult for testing."""
    
    def __init__(self, filing_year, risks):
        self.filing_year = filing_year
        self.risks = risks


class TestZeroRiskHandling:
    """Test that the report handles zero risks gracefully."""
    
    def test_handles_zero_risks(self):
        """Report should explicitly handle case where no risks are detected."""
        results = [MockRiskAnalysisResult(filing_year=2024, risks=[])]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            output_file = f.name
        
        try:
            generate_markdown_report("TEST", results, output_file, filings_db_path=None)
            
            # Read the generated report
            with open(output_file, 'r') as f:
                content = f.read()
            
            # Should explicitly state no risks detected
            assert "no material risks detected" in content.lower() or "no risks detected" in content.lower()
        finally:
            Path(output_file).unlink(missing_ok=True)


class TestRiskOrdering:
    """Test that risks are ordered by novelty, not severity."""
    
    def test_risks_ordered_by_novelty(self):
        """Risks should be ordered by novelty (descending), not severity."""
        # Create risks with different novelty/severity combinations
        risks = [
            {
                "text": "Risk 1: High severity, low novelty. Our operational infrastructure faces severe challenges. " * 50,
                "severity": {"value": 0.9, "explanation": "Very severe"},
                "novelty": {"value": 0.2, "explanation": "Not novel"},
                "category": "OPERATIONAL"
            },
            {
                "text": "Risk 2: Medium severity, high novelty. New regulatory framework creates uncertainty. " * 50,
                "severity": {"value": 0.5, "explanation": "Moderate"},
                "novelty": {"value": 0.8, "explanation": "Very novel"},
                "category": "REGULATORY"
            },
            {
                "text": "Risk 3: Low severity, medium novelty. Financial market conditions are moderately concerning. " * 50,
                "severity": {"value": 0.3, "explanation": "Minor"},
                "novelty": {"value": 0.5, "explanation": "Somewhat novel"},
                "category": "FINANCIAL"
            }
        ]
        
        results = [MockRiskAnalysisResult(filing_year=2024, risks=risks)]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            output_file = f.name
        
        try:
            generate_markdown_report("TEST", results, output_file, filings_db_path=None)
            
            with open(output_file, 'r') as f:
                content = f.read()
            
            # Find the Material Risk Factors section
            material_section_start = content.find("## Material Risk Factors")
            assert material_section_start > 0, "Material Risk Factors section not found"
            
            # Only look within the Material Risk Factors section (next 5000 chars)
            material_section = content[material_section_start:material_section_start + 5000]
            
            # Find positions of each risk in the material section
            # Look for unique text from each risk
            regulatory_pos = material_section.find("regulatory framework")
            financial_pos = material_section.find("Financial market conditions")
            operational_pos = material_section.find("operational infrastructure")
            
            # All should appear (none are -1)
            assert regulatory_pos > 0, "REGULATORY risk not found in Material Risk Factors"
            assert financial_pos > 0, "FINANCIAL risk not found in Material Risk Factors"
            assert operational_pos > 0, "OPERATIONAL risk not found in Material Risk Factors"
            
            # Check order: REGULATORY (novelty 0.8) should come before FINANCIAL (0.5), which should come before OPERATIONAL (0.2)
            assert regulatory_pos < financial_pos, \
                f"REGULATORY {regulatory_pos} should come before FINANCIAL {financial_pos}"
            assert financial_pos < operational_pos, \
                f"FINANCIAL {financial_pos} should come before OPERATIONAL {operational_pos}"
        finally:
            Path(output_file).unlink(missing_ok=True)


class TestCompanyName:
    """Test that company name appears in the report header."""
    
    def test_includes_ticker_in_header(self):
        """Report header should include ticker (since peers table may be empty)."""
        risks = [
            {
                "text": "Sample risk with sufficient length for testing. " * 30,
                "severity": {"value": 0.5, "explanation": "Moderate"},
                "novelty": {"value": 0.5, "explanation": "Somewhat novel"},
                "category": "OPERATIONAL"
            }
        ]
        results = [MockRiskAnalysisResult(filing_year=2024, risks=risks)]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            output_file = f.name
        
        try:
            generate_markdown_report("AAPL", results, output_file, filings_db_path=None)
            
            with open(output_file, 'r') as f:
                first_line = f.readline()
            
            # Should have ticker in header
            assert "AAPL" in first_line
            # Should have the standard header format
            assert "Risk Factor Analysis" in first_line
        finally:
            Path(output_file).unlink(missing_ok=True)


class TestEmojiRemoval:
    """Test that emojis are replaced with ASCII (for PDF compatibility)."""
    
    def test_no_emojis_in_output(self):
        """Report should not contain emojis."""
        risks = [
            {
                "text": "Critical risk requiring immediate attention. " * 40,
                "severity": {"value": 0.85, "explanation": "Critical impact"},
                "novelty": {"value": 0.7, "explanation": "Novel risk"},
                "category": "OPERATIONAL"
            }
        ]
        results = [MockRiskAnalysisResult(filing_year=2024, risks=risks)]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            output_file = f.name
        
        try:
            generate_markdown_report("TEST", results, output_file, filings_db_path=None)
            
            with open(output_file, 'r') as f:
                content = f.read()
            
            # Check for common emojis that were in the old version
            forbidden_chars = ['🔴', '🟠', '🟡', '🟢', '🚨', '✅', '⚠️', '📂', '♻️', '🔍', '★']
            for char in forbidden_chars:
                assert char not in content, f"Found emoji {char} in report"
            
            # Should have ASCII replacements instead
            assert "[CRITICAL]" in content or "[HIGH]" in content or "[MODERATE]" in content
        finally:
            Path(output_file).unlink(missing_ok=True)


class TestLegalDisclaimer:
    """Test that legal disclaimer is included."""
    
    def test_includes_legal_disclaimer(self):
        """Report should include legal disclaimer at bottom."""
        risks = [
            {
                "text": "Sample risk for disclaimer testing. " * 30,
                "severity": {"value": 0.5, "explanation": "Moderate"},
                "novelty": {"value": 0.5, "explanation": "Somewhat novel"},
                "category": "OPERATIONAL"
            }
        ]
        results = [MockRiskAnalysisResult(filing_year=2024, risks=risks)]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            output_file = f.name
        
        try:
            generate_markdown_report("TEST", results, output_file, filings_db_path=None)
            
            with open(output_file, 'r') as f:
                content = f.read()
            
            # Check for key disclaimer phrases
            assert "USE AT YOUR OWN RISK" in content
            assert "informational purposes only" in content
            assert "you may lose money" in content.lower()
            assert "Legal Disclaimer" in content
        finally:
            Path(output_file).unlink(missing_ok=True)


class TestFilingReferenceRemoval:
    """Test that Filing Reference URLs are removed."""
    
    def test_no_filing_reference_urls(self):
        """Filing Reference URLs should be removed (they were incorrect)."""
        risks = [
            {
                "text": "Sample risk for URL testing. " * 30,
                "severity": {"value": 0.5, "explanation": "Moderate"},
                "novelty": {"value": 0.5, "explanation": "Somewhat novel"},
                "category": "OPERATIONAL"
            }
        ]
        results = [MockRiskAnalysisResult(filing_year=2024, risks=risks)]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            output_file = f.name
        
        try:
            generate_markdown_report("TEST", results, output_file, filings_db_path=None)
            
            with open(output_file, 'r') as f:
                content = f.read()
            
            # Should NOT contain the old broken filing reference pattern
            assert "#para1" not in content
            assert "#para2" not in content
            # The old "Filing Reference:" line should not appear in risk cards
            # (It may appear elsewhere like in the header for the actual filing URL)
            lines = content.split('\n')
            risk_section_started = False
            for line in lines:
                if "Material Risk Factors" in line:
                    risk_section_started = True
                # If we're in the risk section and see a risk card, check it
                if risk_section_started and "### Risk #" in line:
                    # Read the next 20 lines (the risk card)
                    idx = lines.index(line)
                    risk_card = '\n'.join(lines[idx:idx+20])
                    # Should not have Filing Reference in risk cards
                    assert "Filing Reference:" not in risk_card or "View in 10-K" not in risk_card
        finally:
            Path(output_file).unlink(missing_ok=True)
