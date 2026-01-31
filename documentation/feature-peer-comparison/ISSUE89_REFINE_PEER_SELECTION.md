Summary
We currently pick micro-cap peers for large companies (example: AAPL compared with SCKT), which produces misleading 100th-percentile severity scores and spurious "novelty" flags caused by trivial formatting/ordering changes. This task refines the RiskAnalyzer comparison layer to (1) validate peer sets by market-cap similarity, (2) compare paragraphs semantically (not just token overlap), and (3) surface "Outlier/Unique Alpha Risks" that are genuinely unique to the target.

Why
Prevent nonsensical peers (micro-cap vs multi-trillion-cap) from skewing percentiles.
Distinguish structural/formatting changes (re-orders, punctuation) from substantive changes in risk content.
Highlight truly unique risks for downstream analysts/investors.
Goals
Implement validate_peer_group(target_ticker: str, peer_tickers: List[str]) -> List[str] using yfinance to filter peers by marketCap (order-of-magnitude rule and fallback to top N by market cap within same SIC).
Replace/augment Jaccard token similarity with a semantic similarity method (sentence-transformers or TF-IDF with larger n-grams) for paragraph-level comparison; treat paragraphs >= 0.9 semantic similarity as non-novel.
Add classification logic to identify "Unique Alpha Risks": target paragraphs with semantic similarity < 0.3 across the entire validated peer set.
Maintain backward compatibility: wrap enhanced logic so EDGAR ingestion and existing flows are not broken; provide config toggles/fallbacks.
Requirements / Spec
1) Peer Set Validation (yfinance)
New helper:
validate_peer_group(target_ticker: str, peer_tickers: List[str]) -> List[str]
Type annotations required.
Behavior:
Query marketCap for target_ticker and each peer in peer_tickers using yfinance (cached where possible).
Filter rule: include peers with marketCap in range [0.1 * target_marketCap, 10 * target_marketCap] (same order of magnitude).
Fallback: if the filtered set is empty or very small, select top 10 by marketCap among peers that share the same SIC (or related industry code), if SIC is available.
If SIC is unavailable, fallback to top 10 by marketCap from peer_tickers.
Return a list of validated tickers (preserve ordering by marketCap descending).
Edge cases:
Missing marketCap (yfinance NA): treat as exclude unless needed for fallback; log warnings.
Very large target marketCap (e.g., > 1T): same rule applies.
Rate limiting: implement simple retry/backoff and local caching for repeated calls during test runs.
2) Semantic vs. Token Similarity
Replace/Augment current token-based Jaccard with a semantic approach.
Suggested API:
compute_semantic_similarity(a: str, b: str) -> float
embed_paragraphs(paragraphs: List[str]) -> np.ndarray
semantic_similarity_matrix(paragraphs: List[str]) -> np.ndarray
Implementation choices:
Preferred: sentence-transformers (small model, e.g., "all-mpnet-base-v2" or a lighter distil model) for sentence embeddings and cosine similarity.
Lighter alternative: scikit-learn TfidfVectorizer with ngram_range=(1,3) and cosine similarity.
Thresholds:
If paragraph similarity >= 0.90 to a paragraph from previous year or peer, do NOT flag as "Novel".
If paragraph similarity < 0.30 across the peer set, consider it a candidate for "Unique Alpha Risk" (see categorization).
Implementation notes:
Use L2-normalized vectors and cosine similarity.
Batch embeddings for large docs.
Provide a config switch to choose backend ("semantic" vs "tfidf") for environments where sentence-transformers are unavailable.
3) Categorization Logic: Outlier / Unique Alpha Risks
New/updated behavior:
For each target paragraph in the 10-K, compute semantic similarity against all paragraphs in each validated peer's 10-K (or peer aggregate).
Compute the maximum similarity per peer paragraph and then compute the peer-group statistic (e.g., median or mean of maxima across peers).
If the peer-group maximum similarity < 0.30 (configurable), label the paragraph as "Unique Alpha Risk".
Suggested API:
identify_unique_alpha_risks(
    target_paragraphs: List[str],
    peer_paragraphs_by_ticker: Dict[str, List[str]],
    similarity_backend: str = "semantic",
    unique_threshold: float = 0.3
) -> List[Dict[str, Any]]
Technical constraints
Type annotations required for all new functions. Code must be mypy-compatible.
Incremental Stability: Do not change EDGAR ingestion or existing parsing pipelines. Wrap the comparison logic (where the current Jaccard is computed) with the new validation and semantic logic so that the rest of the system is unaffected if the new modules are disabled or unavailable.
Provide a safe fallback: if semantic backend fails (no model, out-of-memory), fall back to existing Jaccard/token logic and log a warning.
New dependencies (opt-in):
yfinance
sentence-transformers (preferred)
numpy, scikit-learn (for fallback TF-IDF), scipy
Performance:
Cache marketCaps (TTL e.g., 24 hours) to reduce yfinance calls.
Cache embeddings per (ticker, filing_year) where feasible.
Embed in batches; consider dimension reduction if memory is an issue.
Logging & observability:
Add debug logs for peer validation decisions and number of peers selected.
Add counters for paragraphs flagged as Novel vs Unique Alpha.
Testing
Unit tests:
validate_peer_group: mock yfinance responses; tests for order-of-magnitude filtering, SIC fallback, missing marketCaps.
compute_semantic_similarity: tests with paraphrases (should be high similarity), reorderings (should be high similarity), and genuinely different text (low similarity). Use small sentence-transformers model or TF-IDF in tests to control determinism.
identify_unique_alpha_risks: synthetic peers where one paragraph is unique; assert label assignment.
Integration tests:
Small sample where AAPL-like large cap should not be paired with micro-cap peers; assert validated peer list excludes them.
Perf test: embedding 10 companies' 10-K paragraphs within reasonable time/memory limits.
Mypy checks for new modules.
CI: add optional pipeline job that installs sentence-transformers to run full semantic tests; keep tfidf-only pipeline as fast baseline.
Acceptance criteria
validate_peer_group exists and passes unit tests, correctly filtering peers in all edge cases described.
Semantic similarity method implemented with configurable backend and passes similarity unit tests (paraphrase vs reorder).
identify_unique_alpha_risks flags paragraphs with peer-group similarity < 0.3 and produces expected metadata (per-ticker similarities).
No changes to EDGAR ingestion or other parts of the pipeline unless the enhanced comparison is explicitly enabled.
Code uses type annotations and passes mypy.
Documentation added: short README in RiskAnalyzer module describing new options, thresholds, and fallback behavior.
Suggested function signatures:

from typing import List, Dict, Any

def validate_peer_group(target_ticker: str, peer_tickers: List[str]) -> List[str]:
    ...

def compute_semantic_similarity(a: str, b: str, backend: str = "semantic") -> float:
    ...

def identify_unique_alpha_risks(
    target_paragraphs: List[str],
    peer_paragraphs_by_ticker: Dict[str, List[str]],
    similarity_backend: str = "semantic",
    unique_threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    ...