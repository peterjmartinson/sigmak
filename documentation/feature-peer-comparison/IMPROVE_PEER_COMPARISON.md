📊 Evaluation: Peer Discovery & Risk Comparison System
1. PEER IDENTIFICATION
✅ Strengths:
a) SIC-Based Discovery (peer_discovery.py)

Uses official SEC SIC (Standard Industrial Classification) codes for industry matching
Leverages company_tickers.json (11,000+ companies) and per-company submissions
Caching strategy minimizes SEC API calls (respects rate limits)
Database persistence (peers table in SQLite) for fast repeated queries
b) Robust Implementation:

Handles missing data gracefully (no ticker/CIK/SIC → empty results)
Fallback logic: queries DB first, then on-demand SEC fetches
User-agent header for polite SEC API usage
Configurable top_n for peer count control
c) Test Coverage:

Unit tests with mocked SEC responses (test_peer_discovery.py)
Integration tests verifying CIK→SIC→peers pipeline
CLI demo (demo_peer_discovery.py)
⚠️ Limitations:
Overly Broad SIC Matching:

SIC codes are 4-digit (e.g., 3571 = "Electronic Computers")
AAPL and DELL would match despite vastly different business models
No filtering by market cap, revenue, or sub-industry
No Peer Quality Ranking:

Returns first N matches from SEC data (arbitrary order)
Doesn't prioritize by:
Market cap similarity
Geographic overlap
Revenue/size comparability
Recent filing availability
Scalability Concerns:

refresh_peers_for_ticker() can iterate through 11,000+ companies
Max-fetch limit helps but no incremental/background updates
No caching of "already-searched" SIC codes
Hard-Coded Thresholds:

No configurable SIC-matching granularity (could use 2-digit, 3-digit tiers)
No "strict vs. loose" matching options
2. RISK COMPARISON ANALYSIS
✅ Strengths:
a) Comprehensive Metrics (generate_peer_comparison_report.py)

The system implements all 4 required comparisons from ISSUE #82:

Textual Novelty (YoY):

Sentence-level diff against previous year
Metric: % new sentences (e.g., HURC: 48.6% novel)
Good for detecting material disclosure changes
Peer Similarity (Jaccard):

Token-based overlap (word set intersection/union)
Lower score = more company-specific risks
HURC: 0.187 Jaccard → highly differentiated risk profile
Risk Density & Volume:

Word count + paragraph count vs. peer median
HURC: 1074 words vs. 968.5 peer median (slightly wordier)
Linguistic Tone:

Risk keyword frequency per 1K words
HURC: 11.17 keywords/1K vs. 16.20 peer median (less alarmist language)
b) Severity Percentile Ranking:

Computes average severity across all risks
Ranks target against peers (e.g., HURC at 83rd percentile)
Uses your new sentiment-weighted severity system ✅
c) Unique/Shared Risk Detection:

Jaccard similarity threshold (0.35) for "shared" classification
Identifies company-specific vs. industry-wide risks
Includes snippet previews for manual review
d) Visual Histogram:

10-bin Jaccard distribution for all target-peer risk pairs
Shows clustering patterns (HURC: most risks in 0.1-0.2 range = low overlap)
e) Category Distribution:

Uses LLM-classified risk categories
Target breakdown (HURC: 30% systematic, 30% operational...)
Enables cross-company category divergence analysis
⚠️ Limitations:
Shallow Text Analysis:

Jaccard is bag-of-words (ignores semantics)
"supply chain disruption" vs. "disrupted supply chain" = different tokens
No embedding-based similarity despite having ChromaDB infrastructure
Misses paraphrased risks that are semantically identical
Novelty Detection Weakness:

Sentence splitting is naive (re.split(r"(?<=[.!?])\s+"))
No semantic deduplication (minor rewording = "new" sentence)
Should use embedding similarity for true novelty detection
No Statistical Significance Testing:

Reports percentiles but no confidence intervals
Small peer sets (6 companies) → noisy rankings
No indication if differences are meaningful
Missing Severity Component Analysis:

Reports aggregate severity but not component breakdown
Your new system tracks sentiment/quant/keywords/novelty but this isn't surfaced in peer comparison
Could show: "HURC has lower quant_anchor scores vs. peers (smaller $ amounts)"
Hard-Coded Keyword List:

Only 9 keywords (your severity module has 22 severe + 14 moderate)
Not reusing existing infrastructure
No Time-Series Comparison:

YoY novelty for target only
Doesn't show how peer novelty evolved over same period
Can't detect if entire industry is experiencing new risks
Weak Shared Risk Reporting:

Threshold at 0.35 Jaccard is arbitrary
HURC report shows "no shared risks" despite having 6 peers
Should use embedding similarity or lower threshold
3. OUTPUT QUALITY
✅ Good:
Clean Markdown format with sections/bullets
Example reports exist (AAPL_Peer_Comparison_2025.md, HURC_Peer_Comparison_2025.md)
Includes histogram visualization (ASCII art)
Handles missing peer data gracefully
⚠️ Issues:
AAPL report shows "no peers analyzed" (discovery failed or peers not downloaded)
"Specific comparisons" note says "simple token-based metrics" → acknowledges weakness
No risk severity histogram (only Jaccard similarity)
Missing actionable insights (just raw numbers, no interpretation)
4. INTEGRATION WITH NEW SEVERITY SYSTEM
✅ Partially Integrated:
Peer comparison uses compute_severity_avg() which calls your new scorer
Percentile rankings reflect sentiment-weighted severity
❌ Not Leveraging Full Potential:
No component breakdown (sentiment vs. quant vs. keywords vs. novelty)
No novelty_score usage (peer comparison computes YoY novelty separately using text diffs, ignores your embedding-based novelty)
Dollar amount extraction unused in peer comparison (could compare $ magnitudes across companies)
Market cap normalization unused in peer reports
5. RECOMMENDATIONS
High Priority:
Use Embedding Similarity for Shared Risks:

Surface Severity Components in Peer Report:

Add table showing average sentiment/quant/keyword/novelty scores per company
Highlight: "Target has 2x higher quant_anchor scores (larger $ risks)"
Add Market Cap Context:

Show peer list with market caps
Rank peers by cap proximity to target
Normalize severity metrics by company size
Improve Peer Selection:

Filter by market cap range (±50% of target)
Prioritize companies with recent filings
Add SIC similarity tiers (4-digit strict, 3-digit loose)
Medium Priority:
Statistical Significance:

Add confidence intervals for percentile rankings
Flag when peer sample is too small (<5 companies)
Show z-scores vs. peer mean
Semantic Novelty Detection:

Use your ChromaDB novelty scoring for YoY comparison
Report embedding distance (not just sentence-level diffs)
Risk Category Divergence:

Compute KL-divergence between target/peer category distributions
Highlight unusual category concentrations
Low Priority:
Historical Peer Benchmarking:

Compare target's YoY trend vs. peer YoY trends
Detect if target is diverging from industry
Interactive Visualization:

Generate CSV for external visualization
Add scatter plots (severity vs. novelty, target vs. peers)
FINAL VERDICT:
Aspect	Score	Notes
Peer Discovery	7/10	Works but too broad (SIC-only), no quality ranking
Risk Comparison	6/10	Implements all 4 metrics but shallow (Jaccard, not embeddings)
Integration with Severity v2	4/10	Uses aggregate severity, ignores component breakdown
Output Quality	7/10	Clean reports but missing insights/visualizations
Test Coverage	8/10	Good unit tests, mocked SEC calls
Actionability	5/10	Numbers without interpretation, no "so what?"
Overall: 6.2/10 — Functional but underutilized. The peer comparison is doing basic text analysis when you have a sophisticated embedding/sentiment/quantitative infrastructure that isn't being leveraged.