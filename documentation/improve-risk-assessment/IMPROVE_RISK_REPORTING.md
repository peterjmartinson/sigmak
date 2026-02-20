Summary
The YOY (year-over-year) risk report for individual companies requires multiple fixes and refinements. Please address all points below in the next PR to improve quality, accuracy, and usefulness of the report.

Problems to Fix

1. Intro Section Detected as a Risk

The report almost always extracts the title/introduction of Item 1A as one of the risks. This is not an actual risk, but a boilerplate intro.
Action: Add logic to filter out or ignore this section.
File references: scripts/generate_yoy_report.py, src/sigmak/integration.py
2. Incorrect Filing Reference Section

The “Filing Reference:” in risk listings is wrong: it always generates BASEURL/para1, BASEURL/para2, following the risk numbering, regardless of the real paragraph.
Action: Fix the logic so that Filing Reference correctly matches the real source of the risk text, or omit it until accurate.
File references: scripts/generate_yoy_report.py and any underlying reference helpers
3. Severity Based on Boilerplate

Severity scoring sometimes reflects generic/boilerplate language instead of company-specific detail.
Action: Plan improvements to separate boilerplate from firm-specific signals and triage worst examples.
File references: src/sigmak/integration.py
4. Always Listing 10 Risks (Even If Less Exist)

The report always lists 10 risks, causing non-risks to be included. It’s acceptable to list zero risks and say “no real risks detected.”
Action: Update logic so the system only lists detected risks (can be zero).
File references: scripts/generate_yoy_report.py
5. Novelty Should Determine Risk Order

Risks should be ordered by novelty vs previous years (most novel first), not by severity.
Action: Sort reported risks by novelty.
File references: scripts/generate_yoy_report.py, scripts/generate_peer_comparison_report.py
6. Simple Presentation Issues
A. Include company name (not just ticker) at the top of the report (use DB table that maps tickers → names).
B. Replace emojis (severity indicators, etc.) with ASCII or histogram representations for PDF compatibility.
C. Add a legal disclaimer at the bottom: Use at your own risk. This is for informational purposes only. We cannot guarantee investment returns; you may lose money.

File references: scripts/generate_yoy_report.py, scripts/generate_peer_comparison_report.py, src/sigmak/peer_comparison.py
Acceptance Criteria

Item 1A intro and other non-risk boilerplate no longer appear in the list of detected risks.
Filing Reference is accurate, or omitted if it cannot be made accurate.
If no real risks are found, the report explicitly states this (zero-risk case handled).
Risks are ordered by novelty, not severity.
Company name is shown in the report.
All emojis are replaced with ASCII/histogram equivalents.
Legal disclaimer text appears at the bottom.