Problem
Current Item 1A (Risk Factors) extraction relies on regex/text heuristics after flattening the document to plain text. That approach fails on filings where section boundaries are non-standard, headings are split/obfuscated, or formatting varies widely. We previously tried edgartools and regex; both have pitfalls on some filings.

Proposal summary
Use the Python package sec-parser as the DOM-first extractor for Item 1A by default. If sec-parser extraction fails (package missing, parser exception, or implausibly short/irrelevant result), fall back to the existing regex-based extractor. Make this behavior the default for all ingestion entrypoints (e.g., generate_yoy_report.py and any other scripts that trigger the ingestion pipeline).

Important: the repository uses the "uv" tool for dependency installs — document the installation step for sec-parser using uv rather than pip.

Goals / requirements
Default behavior: try sec-parser first; on failure, run the existing fallback extractor.
All ingestion entrypoints (scripts and pipeline methods that perform ingestion/indexing/analysis) must use sec-parser-first flow by default.
Integration must be defensive and non-blocking: if sec-parser is missing or raises errors, pipeline must continue using existing logic.
Log loudly (ERROR-level) when sec-parser is not installed or cannot be imported; message should include next steps/instructions to install via uv (e.g., "sec-parser not installed — falling back to regex. To enable sec-parser run: uv add sec-parser").
Make the integration configurable:
ENV: ENABLE_SEC_PARSER=true|false (default true)
Optionally DOM_EXTRACTOR=sec-parser|regex for explicit control
Use robust encoding handling when reading local .htm/.html files (keep the existing UTF-8/CP1252 fallbacks and consider chardet if needed).
Installation note (uv)
To add the dependency locally / CI, use your existing "uv" workflow rather than pip. Document the uv command in README / CONTRIBUTING:
Example: uv add sec-parser or add sec-parser to your project dependency file and run uv install per your repo convention.
Keep sec-parser an optional dependency (try/except ImportError) so installations without it continue to work.
Logging & diagnostics
Log which extraction path was used for each filing: "sec-parser:success", "sec-parser:short_result", "sec-parser:error", or "fallback:regex".
If sec-parser import fails, log an ERROR-level message with install instructions and then fallback.
Persist a small diagnostics sample set (timestamped HTML + extracted snippet + extraction path) for failing or ambiguous files to a diagnostics directory for triage.
Tests & benchmarking
Ground-truth set: gather 20–50 representative 10‑K .htm files (diverse issuers/years/formatting).
For each file:
Confirm "Item 1A" exists in raw HTML.
Run sec-parser-first extractor and the existing regex extractor.
Manually mark start/end for a subset to compute precision/recall and common failure modes.
Add automated unit tests:
Mock sec-parser to simulate success/failure for edge cases.
Integration tests that assert the pipeline uses sec-parser when enabled and falls back when it returns None/raises.
Acceptance criteria:
When sec-parser is installed and ENABLE_SEC_PARSER is true (default), the pipeline uses sec-parser-first flow unless its result is None/short/invalid.
No regressions for files that previously worked with regex-only extractor.
Implementation considerations
Make sec-parser optional: avoid hard dependency failures at runtime. Use try/except ImportError and fallback behavior.
Config toggles:
ENV: ENABLE_SEC_PARSER=true|false (default true)
Optionally DOM_EXTRACTOR=sec-parser|regex for explicit control.
Update ingestion entrypoints (generate_yoy_report.py, analyze_filing.py, scripts, and pipeline methods) to prefer the pipeline default — do not change other repository structure without prior approval.
Ensure file reading uses robust encoding detection and optional HTML cleanup (strip scripts/styles) if needed.
The assistant/agent may produce a non-invasive patch implementing the spec on request; no code will be committed without explicit approval.
Next actions (no PR; issue only)
This issue requests a small, defensive integration and tests. No PR will be created unless explicitly asked.
If you want, I can produce a suggested patch (diff) for review before any commit.
Identified ingestion entrypoints (reviewed)

scripts/generate_yoy_report.py — top-level report generator that uses IntegrationPipeline.
scripts/analyze_filing.py — CLI analyzer that instantiates IntegrationPipeline and runs pipeline.analyze_filing(...).
src/sigmak/integration.py — IntegrationPipeline; orchestrates ingestion and indexing (update pipeline default here).
src/sigmak/indexing_pipeline.py — IndexingPipeline.index_filing calls extract_text_from_file() and slice_risk_factors(); update index_filing to call the sec-parser-first extractor before falling back.
src/sigmak/ingest.py — current text extraction and slice_risk_factors() live here; integrate sec-parser fallback logic here or add a DOM-first wrapper that the pipeline calls.
tests/test_ingestion.py — add tests that mock sec-parser present/missing and validate fallback behavior.
Formatting & metadata

Default environment variable: ENABLE_SEC_PARSER=true.