# Add BOILERPLATE Classification Category

**Date**: 2026-02-14  
**Issue**: TOC and intro text appearing in risk analysis results  
**Solution**: Replace regex pattern matching with semantic LLM classification

---

## Problem Statement

Current regex-based filtering fails to catch all boilerplate variations:
- Table of Contents lines (e.g., "PagePART I.Item 1.Business1Item 1A.Risk Factors9...")
- Generic intro paragraphs (e.g., "The risks described below could...")
- Section headers without content
- Filing metadata

**Example**: ABT 2023 filing has TOC line appearing as Risk #4 in report despite two layers of regex filtering.

Even the LLM recognizes it in its rationale:
> "The provided text is a table of contents for an SEC filing, **not an actual risk disclosure** from Item 1A."

---

## Proposed Solution

### Architecture: Classification-Based Filtering

Replace brittle regex patterns with semantic understanding:

1. **Add BOILERPLATE category** to classification system
2. **Simplify regex** to basic sanity checks only
3. **LLM classifies** boilerplate during normal classification flow
4. **Filter BOILERPLATE** from reports before display
5. **Cache in vector DB** for future similarity matching (self-improving)

### Key Insight

**Use regex for syntax, use LLM for semantics**

- Regex: Fast checks for obvious non-text (empty, no punctuation)
- LLM: Semantic understanding of "is this an actual risk disclosure?"

---

## Implementation Plan

### 1. Add BOILERPLATE Category

**File**: `src/sigmak/risk_category.py` (or equivalent enum location)

**Change**: Add to `RiskCategory` enum
```python
class RiskCategory(Enum):
    OPERATIONAL = "operational"
    REGULATORY = "regulatory"
    FINANCIAL = "financial"
    SYSTEMATIC = "systematic"
    GEOPOLITICAL = "geopolitical"
    BOILERPLATE = "boilerplate"  # NEW: Non-risk text
    UNCATEGORIZED = "UNCATEGORIZED"
```

**Impact**: Classification system can now label non-risk content

---

### 2. Update Classification Prompt

**File**: `src/sigmak/llm_classifier.py` or `prompts/risk_classification_v1.txt`

**Change**: Add BOILERPLATE definition to system prompt

```
Risk Categories:

- operational: Risks related to internal business operations, supply chain, 
  technology systems, product quality, workforce management, or execution failures

- regulatory: Risks related to government regulation, compliance requirements, 
  legal proceedings, or changes in laws/policies

- financial: Risks related to capital structure, liquidity, credit, interest rates,
  currency fluctuations, or accounting practices

- systematic: Risks related to macroeconomic conditions, market-wide forces,
  economic cycles, inflation, or broad industry trends

- geopolitical: Risks related to international conflicts, trade wars, sanctions,
  terrorism, political instability, or diplomatic relations

- boilerplate: Table of contents, section headers, page numbers, filing metadata,
  generic introductory text (e.g., "The risks described below..."), or any text 
  that is NOT an actual risk disclosure from Item 1A. This includes navigational 
  elements, document structure text, and placeholder content.

IMPORTANT: If the text is not describing a specific risk to the business, 
classify it as BOILERPLATE.
```

**Impact**: LLM will recognize and classify non-risk text

---

### 3. Simplify Solution 1 (Pre-Chunking Filter)

**File**: `src/sigmak/processing.py` - `_strip_item_1a_header()`

**Current**: 35 lines with section header detection and intro paragraph skipping

**Simplified**:
```python
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
```

**Removed**:
- Section header regex pattern (`\n([A-Z][A-Za-z\s,]+Risks?)\n`)
- Intro paragraph skipping logic
- Debug logging about characters removed
- Fallback logic

**Why**: Let BOILERPLATE classification catch intro paragraphs semantically

**Lines**: 35 → 8 lines

---

### 4. Simplify Solution 2 (Post-Retrieval Filter)

**File**: `src/sigmak/processing.py` - `is_valid_risk_chunk()`

**Current**: 35 lines with 6+ pattern checks

**Simplified**:
```python
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
```

**Removed**:
- Item 1A title pattern matching
- TOC dot pattern (`\.{3,}`)
- Page number pattern (`\bpage\s+\d+`)
- Generic intro statement patterns
- Length-based heuristics for special cases

**Why**: BOILERPLATE category handles semantic detection better

**Lines**: 35 → 20 lines

---

### 5. Filter BOILERPLATE from Reports

**File**: `scripts/generate_yoy_report.py`

**Location**: In `generate_markdown_report()` function (around line 565)

**Change**: Add filtering before sorting risks

```python
def generate_markdown_report(
    ticker: str,
    results: List["RiskAnalysisResult"],
    output_file: str = "risk_analysis_report.md",
    filings_db_path: str | None = None,
) -> None:
    """Generate investment-grade markdown report focusing on latest year's risks."""
    results_sorted = sorted(results, key=lambda r: r.filing_year)
    years = [r.filing_year for r in results_sorted]
    latest_result = results_sorted[-1]
    latest_year = latest_result.filing_year
    changes = identify_risk_changes(results_sorted)
    
    # Filter out boilerplate before sorting
    valid_risks = [
        r for r in latest_result.risks 
        if r.get('category', '').lower() != 'boilerplate'
    ]
    
    boilerplate_count = len(latest_result.risks) - len(valid_risks)
    if boilerplate_count > 0:
        logger.info(f"Filtered {boilerplate_count} boilerplate chunks from {ticker} {latest_year}")
    
    # Sort valid risks by severity (existing logic continues with valid_risks)
    valid_risks = sorted(
        valid_risks,
        key=lambda x: x.get('severity', {}).get('value', 0),
        reverse=True
    )
    
    # ... continue with valid_risks instead of latest_result.risks ...
```

**Impact**: Boilerplate won't appear in final markdown reports

---

### 6. Add Logging and Observability

**File**: `scripts/generate_yoy_report.py`

**Multiple locations**: Add logging throughout pipeline

```python
# After analyze_filing completes
boilerplate_risks = [r for r in result.risks if r['category'] == 'boilerplate']
if boilerplate_risks:
    print(f"   ⚠️  Detected {len(boilerplate_risks)} boilerplate chunks")
    for bp in boilerplate_risks:
        print(f"      • {bp['text'][:80]}...")

# In report generation
print(f"\n📊 Risk Analysis Summary for {ticker} {latest_year}:")
print(f"   • Total chunks retrieved: {len(latest_result.risks)}")
print(f"   • Substantive risks: {len(valid_risks)}")
print(f"   • Boilerplate filtered: {boilerplate_count}")
```

**Impact**: Visibility into filtering effectiveness

---

### 7. Update Tests

**File**: `tests/test_processing.py`

**Changes**:

1. **Simplify header stripping test**:
```python
def test_strip_item_1a_header():
    """Test that Item 1A title is removed (intro detection moved to LLM)."""
    sample_text = """ITEM 1A. RISK FACTORS
The risks described below could materially affect our business.
Strategic Risks
Failure to execute our strategy may harm results."""
    
    stripped = _strip_item_1a_header(sample_text)
    
    # Title should be removed
    assert "ITEM 1A" not in stripped.upper()
    
    # Everything else preserved (intro detection now via LLM)
    assert "The risks described below" in stripped
    assert "Strategic Risks" in stripped
```

2. **Update chunk validation test**:
```python
def test_is_valid_risk_chunk_basic_sanity():
    """Test basic sanity checks only (semantic detection via LLM)."""
    
    # Valid: Has words and punctuation
    assert is_valid_risk_chunk("This is a risk with at least thirty words in a sentence. " * 3)
    
    # Invalid: Too short
    assert not is_valid_risk_chunk("Short text.")
    
    # Invalid: No punctuation
    assert not is_valid_risk_chunk("just keywords no sentences here" * 10)
    
    # Invalid: All caps
    assert not is_valid_risk_chunk("THIS IS ALL CAPS SCREAMING TEXT " * 20)
```

3. **Add BOILERPLATE category test**:
```python
def test_boilerplate_classification(integration_pipeline):
    """Test that TOC and intro text get classified as BOILERPLATE."""
    
    toc_text = "PagePART I.Item 1.Business1Item 1A.Risk Factors9Item 1B..."
    intro_text = "The risks described below could, in ways we may or may not be able to predict..."
    
    # Classify via pipeline
    result_toc = integration_pipeline.llm_classifier.classify(toc_text, use_cache=False)
    result_intro = integration_pipeline.llm_classifier.classify(intro_text, use_cache=False)
    
    assert result_toc.category.value == "boilerplate"
    assert result_intro.category.value == "boilerplate"
```

---

## Processing Flow Change

### Before (Regex-Heavy)

```
Extract Item 1A 
  ↓
[Regex Layer 1] Strip title + section headers + intro paragraph
  ↓ (complex pattern matching)
Chunk text
  ↓
[Regex Layer 2] Validate each chunk:
  - Check Item 1A patterns
  - Check TOC dots
  - Check page numbers
  - Check intro statements
  - Length heuristics
  ↓ (6+ pattern checks per chunk)
Embed & Index to ChromaDB
  ↓
Semantic Search retrieves top-k
  ↓
Score & Classify risks
  ↓
Report (whatever got through regex)
```

**Problems**:
- Regex breaks on format variations
- Adding patterns increases maintenance burden
- False positives/negatives from heuristics
- No learning from past encounters

### After (LLM-Semantic)

```
Extract Item 1A
  ↓
[Regex Layer 1] Strip title only (simple, fast)
  ↓
Chunk text
  ↓
[Regex Layer 2] Basic sanity check:
  - Has words?
  - Has punctuation?
  - Not all-caps?
  ↓ (3 simple checks)
Embed & Index to ChromaDB
  ↓
Semantic Search retrieves top-k
  ↓
[LLM Classification] Categorize including BOILERPLATE detection
  ↓ (semantic understanding + vector cache)
[Filter] Remove category == 'boilerplate'
  ↓
Report (clean results)
```

**Benefits**:
- LLM understands semantic meaning
- Self-improving via vector cache
- Handles format variations naturally
- Simple, maintainable code

---

## Expected Outcomes

### Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Regex patterns | ~8 complex | 2 simple | -75% |
| Lines of validation code | ~70 | ~25 | -64% |
| Boilerplate detection coverage | ~85% | ~98% | +13% |
| False positives in reports | 1-2/filing | ~0/filing | -100% |
| Maintenance burden | High | Low | Major ↓ |
| Self-improving | No | Yes | ✅ |
| LLM cost (first time) | $0 | ~$0.001 | Minimal |
| LLM cost (cached) | $0 | $0 | None |

### Code Reduction

- **processing.py**: 70 lines → 28 lines (-42 lines)
- **Complexity**: High → Low
- **Maintainability**: Difficult → Easy

---

## Testing Strategy

### Unit Tests

1. **Test simplified header stripping**
   - Verify title removal
   - Verify intro text preserved (for LLM to classify)

2. **Test simplified validation**
   - Verify basic sanity checks
   - Remove pattern-specific tests

3. **Test BOILERPLATE classification**
   - TOC text → classified as boilerplate
   - Intro text → classified as boilerplate
   - Real risk → not classified as boilerplate

### Integration Tests

1. **ABT 2023 Filing** (has TOC issue)
   - Process filing
   - Verify TOC classified as BOILERPLATE
   - Verify BOILERPLATE filtered from report
   - Verify substantive risks still present

2. **WMT/BA/VMC Filings** (regression)
   - Re-process existing filings
   - Verify no substantive risks lost
   - Verify results quality maintained

3. **Cache Test**
   - Process same filing twice
   - Verify second run uses vector cache
   - Verify no LLM call on second run

### Manual Verification

1. Check `results_ABT_2023.json`:
   - Contains BOILERPLATE risks with classification
   - LLM rationale explains why it's boilerplate

2. Check `ABT_YoY_Risk_Analysis_2023_2025.md`:
   - No boilerplate in final report
   - Only substantive risks displayed

3. Check logs:
   - "Filtered X boilerplate chunks" messages
   - Classification method shows cache hits

---

## Risks & Mitigations

### Risk 1: LLM Misclassifies Real Risk as Boilerplate

**Likelihood**: Low  
**Impact**: High (loses real risk data)

**Mitigation**:
- Carefully craft prompt with examples
- Test on diverse filings before rollout
- Monitor classification rationales in logs
- Keep simplified regex as first-pass filter

**Rollback**: Revert BOILERPLATE filtering, inspect misclassifications

### Risk 2: Performance Degradation from LLM Calls

**Likelihood**: Low  
**Impact**: Medium (slower processing)

**Mitigation**:
- Vector cache hits after first classification
- Most TOCs/intros similar across filings
- Layer 1 still prevents obvious junk from being embedded

**Monitoring**: Track cache hit rate, LLM call counts

### Risk 3: Breaking Existing Tests

**Likelihood**: Medium  
**Impact**: Low (caught before merge)

**Mitigation**:
- Update tests alongside code changes
- Run full test suite before commit
- Keep test coverage high

**Fix**: Update test expectations to match new behavior

### Risk 4: Unexpected Boilerplate Patterns

**Likelihood**: Medium  
**Impact**: Low (self-correcting)

**Mitigation**:
- System learns from each new pattern
- Cache builds over time
- LLM handles semantic variations

**Monitoring**: Track new boilerplate classifications in logs

---

## Rollback Plan

If critical issues arise:

### Step 1: Emergency Rollback (< 5 min)
```bash
git revert <commit-hash>
```

### Step 2: Partial Rollback (< 30 min)

Keep simplifications, remove BOILERPLATE filtering:

1. Comment out BOILERPLATE filtering in `generate_yoy_report.py`
2. Let boilerplate get classified but still appear in reports
3. Investigate misclassifications
4. Fix prompt and re-enable filtering

### Step 3: Full Rollback (< 2 hours)

1. Revert `RiskCategory` enum changes
2. Revert classification prompt updates
3. Restore complex regex patterns
4. Revert test changes

**Note**: Keep simplified regex even in rollback - it's objectively better

---

## Success Criteria

### Must Have
- ✅ ABT TOC line classified as BOILERPLATE
- ✅ ABT report contains 0 boilerplate risks
- ✅ All existing integration tests pass
- ✅ No substantive risks lost in WMT/BA/VMC filings

### Should Have
- ✅ Code reduced by 40+ lines
- ✅ Cache hit rate > 80% on second filing
- ✅ Processing time within 10% of current

### Nice to Have
- ✅ Future boilerplate variations caught automatically
- ✅ Observability dashboard showing boilerplate patterns
- ✅ Zero manual pattern additions needed in next 6 months

---

## Timeline

1. **Implement** (2-3 hours)
   - Add BOILERPLATE category
   - Update prompt
   - Simplify regex
   - Add filtering
   - Update tests

2. **Test** (1-2 hours)
   - Unit tests
   - Integration tests
   - Manual verification

3. **Deploy** (30 min)
   - Commit changes
   - Update documentation
   - Monitor first runs

4. **Monitor** (1 week)
   - Check classification quality
   - Track cache hit rates
   - Gather feedback

---

## Future Enhancements

### Phase 2: Analytics Dashboard
- Query: "Show all unique boilerplate patterns across 500 filings"
- Metric: "Boilerplate detection rate by company/year"
- Insight: "Common filing format changes over time"

### Phase 3: Active Learning
- Flag low-confidence BOILERPLATE classifications for review
- Human feedback improves future classifications
- Build training dataset for fine-tuning

### Phase 4: Multi-Category Filtering
- Add DUPLICATE category for repetitive risks
- Add IMMATERIAL category for low-impact disclosures
- Configurable filtering in reports

---

## References

- Original Issue: TOC appearing in ABT 2023 report (risk #4)
- Related: JOURNAL.md entry [2026-02-13] Filter Item 1A Boilerplate
- Code: `src/sigmak/processing.py`, `src/sigmak/integration.py`
- Tests: `tests/test_processing.py`, `tests/test_integration_pipeline.py`

---

**Status**: Ready for Implementation  
**Approval Required**: Yes  
**Estimated Effort**: 4-6 hours  
**Risk Level**: Low-Medium
