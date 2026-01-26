Batch Analyze Target & Peer Filings, Generate Markdown Peer Comparison Report #82

Objective
Implement a Python script that, given a target ticker and year, analyzes risk factors for:

the target company (e.g., NVDA 2024)
its industry peers (any subset with available downloaded filings)
and generates a Markdown report benchmarking the target's risk profile vs. peers.
Key Requirements
CLI utility (scripts/generate_peer_comparison_report.py) that:
Accepts target ticker and year (and optionally peer tickers)
Loads filings for target and peers from the previous download step
Runs the full analysis pipeline (IntegrationPipeline) for each filing
Computes peer comparison stats: severity/novelty percentiles, category divergences, unique/shared risks.
Generates a Markdown report (in output/) showing how the target company compares to its industry peers (risk profile, risk factor distribution, stand-out risks, etc.).
Modular design: independent of downloading/discovery logic (can load any files present)
Logs/prints summary of what was analyzed and any missing filings
Tests for logic using small synthetic/mock filings in tests/data/
Acceptance Criteria
Stand-alone workflow: Run script after discovery + download, get a report with peer benchmarking
Handles subset analysis when some peers have no filing
Clean, actionable, easy-to-copy Markdown output suitable for investors or technical review

Specific items for comparison:

Textual Novelty (Year-over-Year):

Compare the current 10-K risk section to the previous year’s 10-K for the same company.

Metric: Percentage of "New" vs. "Repeated" sentences.

Peer Similarity Score:

Calculate the Jaccard Similarity or Cosine Similarity between the target's risk section and the industry peer average.

Insight: A low similarity score indicates "Company-Specific" risks that aren't industry-standard.

Risk Density & Volume:

Metric: Word count and paragraph count relative to the peer median.

Linguistic Tone (Sentiment):

Count occurrences of financial "Risk" keywords (e.g., litigation, volatility, uncertainty, adverse).

Metric: Keywords per 1,000 words compared to peers.