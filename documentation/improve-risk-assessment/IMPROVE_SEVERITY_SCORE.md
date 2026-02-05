Title: Refactor severity scoring: sentiment-weighted, quantitative-anchor, market-cap normalized (replace legacy scorer)

Body:
Overview
Refactor the risk severity scoring in sigmak to use a weighted, sentiment-first approach that integrates quantitative anchors (e.g., debt size), risk keyword counts, and textual novelty, normalized by market cap when possible.

Goal
Improve precision of risk identification (Item 1A, SEC filings) by replacing the existing scorer with a configurable, explainable severity calculation that uses sentiment, monetary anchors, keyword density, and novelty—normalized against the company's market cap (from SQLite DB).

Implementation Tasks
1. Design & Specs

Document scoring formula and component features:
severity = w_sentiment*sentiment_score + w_quant*quant_anchor_score + w_count*keyword_count_score + w_novelty*novelty_score
Default weights: sentiment 0.45, quant_anchor 0.35, keyword_count 0.10, novelty 0.10
Quant score: try to extract monetary amounts from text and normalize by market cap (from DB); fallback to log normalization if cap unavailable
Sentiment: use VADER or spaCy/TextBlob for sentence-level scoring over risk-relevant spans
Keyword density: per-1K words, normalized (default divisor: 20)
Novelty: use vector database (Chroma) similarity; novelty = 1.0 - similarity

Store all parameters in config for tuning
2. Module Creation

Create src/sigmak/severity.py with functions:
extract_numeric_anchors(text)
compute_sentiment_score(text, sentence_weights)
compute_quant_anchor_score(amounts, market_cap)
compute_keyword_count_score(text, keyword_list)
compute_novelty_score(embedding, drift_system)
compute_severity(...) → (value, explanation_dict)

Add robust tests to tests/test_severity.py
3. Pipeline Integration

Refactor the scorer in scripts/analyze_filing.py to call the new compute_severity logic for each risk.

Ensure that explanations for each risk (which component contributed most, extracted anchors, etc.) are persisted in output JSON.

Update the pipeline to load market cap from SQLite for each ticker (fallback to log-normal when missing).
4. Dependencies

Add/upgrade spaCy (pref: en_core_web_sm or en_core_web_trf), VADER (vaderSentiment), and any needed NLP deps to requirements.txt
5. Configuration

Store default weights, normalization constants, and keyword lexicon in a YAML config, overridable by env
6. Testing & Validation

Unit test: edge cases (no anchors, multiple amounts, large values, neutral sentiment, high novelty)

E2E test: verify old vs new scorer on sample filings; document observed differences
7. Docs & Checklist

Add/Update documentation for the new severity system and adjustment workflow

Changelog entry

README.md: update example(s), scoring explanation, and usage

JOURNAL.md: add dev log entry for risk scorer v2

PR-ready: CI green, all new code tested
Acceptance Criteria

Risk severity scoring uses the new weighted system with explainable outputs

Market cap normalization is used for quantitative anchors whenever possible

Dependencies and docs are updated; all tests pass
References & Context
Requirements and scoring model as discussed in issue
Example input/output available (attach sample risk block and JSON output)
SEC Item 1A sample HTML available for development/testing


CRITICAL REFINEMENT

In the Boeing text, they mention $4.9B loss (2025) and $3.5B loss (2024). My Advice: Ensure your extract_numeric_anchors logic is instructed to prioritize the maximum value or the value associated with the most recent date. You don't want the model to average them and dilute the current year's "bleed."