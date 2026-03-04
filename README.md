# SigmaK: Risk Intelligence for SEC Filings

[![CI](https://github.com/peterjmartinson/sigmak/actions/workflows/ci.yml/badge.svg)](https://github.com/peterjmartinson/sigmak/actions/workflows/ci.yml)

**SigmaK quantifies how much a company\'s risk picture changed year-over-year — using SEC 10-K filings and semantic embeddings, not keyword search.**

When a company\'s risk language shifts, that signal is buried in thousands of words of boilerplate. SigmaK extracts Item 1A, scores each paragraph for *severity* and *novelty*, classifies it into a proprietary risk taxonomy, and surfaces what\'s actually new — across years and across peer companies.

---

## Why SigmaK?

| | Keyword Search | SigmaK |
|---|---|---|
| **Novelty Detection** | Finds words, misses meaning | Semantic delta: scores how different new disclosures are from historical baselines |
| **Risk Classification** | Ad-hoc text matching | Proprietary 10-category taxonomy, LLM-backed with vector caching |
| **Drift Tracking** | Manual review | Automated year-over-year comparison with confidence scoring |
| **Peer Context** | None | Benchmarks severity percentile against SIC-matched peer companies |
| **Provenance** | None | Every score includes method, model version, confidence, and source citation |

---

## Sample Output — ABT (Abbott Laboratories) FY2025

*Generated from `results_ABT_2025.json` and `ABT_Peer_Comparison_2025.yaml`*

### Risk Factor Scores (Item 1A)

| Category | Risk (excerpt) | Severity | Novelty | Confidence |
|---|---|---|---|---|
| OPERATIONAL | Global supply chain disruptions could negatively affect results of operations | 0.63 | 0.00 | 98% |
| GEOPOLITICAL | International ops: tariffs, trade sanctions, expropriation, sovereign debt | 0.62 | 0.00 | 90% |
| GEOPOLITICAL | Russia/Ukraine: sanctions, economic volatility, supply disruptions | 0.61 | 0.00 | 95% |
| REGULATORY | Cost of compliance with governmental regulations; non-compliance is material | 0.50 | 0.00 | 98% |
| FINANCIAL | FX exposure; government exchange controls; hedging uncertainty | 0.36 | 0.33 | 98% |

### Peer Comparison (vs. MDT, JNJ, BSX, SYK, DHR)

| Metric | ABT | Peer Median |
|---|---|---|
| Avg Severity | 0.518 | — |
| Severity Percentile | **80th** | — |
| Risk Paragraphs | 9 | 7 |
| Linguistic Intensity¹ (per 1k words) | 19.1 | 14.6 |
| Textual Novelty (YoY new sentences) | **72.2%** | — |
| Peer Similarity (avg Jaccard) | 0.25 | — |

> ¹ **Linguistic Intensity** = frequency of high-impact risk adjectives and escalatory language (e.g., *material*, *adverse*, *catastrophic*, *could have a significant effect*) per 1,000 words. Higher values indicate more emphatic risk disclosure language.

> **Reading the scores:** Severity (0–1) = potential business impact. Novelty (0–1) = how semantically different this disclosure is from the company\'s historical baseline. `0.00` novelty = language repeated verbatim from prior filings. `0.33` novelty on the FX risk = that language has shifted meaningfully year-over-year.

---

## Key Features

- **Semantic Risk Scoring**: Severity (0–1) and Novelty (0–1) for every Item 1A paragraph, with source citation and explanation
- **Proprietary Risk Taxonomy**: 10-category classification (see below) backed by versioned LLM prompts and vector caching
- **Threshold-Based Routing**: Auto-selects vector search, LLM confirmation, or full LLM classification based on confidence; falls back to Gemini 2.5 Flash Lite for low-confidence cases
- **Year-over-Year Analysis**: Compares risk language across multiple 10-K filings for the same company
- **Peer Benchmarking**: Scores a company against SIC-matched peers using Jaccard similarity, severity percentile, and linguistic intensity
- **Drift Detection**: Periodic re-evaluation of stored classifications to catch model decay over time
- **Full Provenance**: Every output record carries method, model version, confidence, prompt version, and timestamp
- **Async REST API**: FastAPI + Celery + Redis for production non-blocking deployments

---

## Risk Taxonomy

SigmaK classifies every risk paragraph into one of 10 proprietary categories:

| Category | Description | Examples |
|---|---|---|
| **OPERATIONAL** | Internal execution risks | Supply chain, manufacturing, IT infrastructure |
| **SYSTEMATIC** | Macroeconomic forces | Recession, inflation, market volatility |
| **GEOPOLITICAL** | International conflicts | War, trade disputes, sanctions |
| **REGULATORY** | Government compliance | New laws, regulations, policy changes |
| **COMPETITIVE** | Market rivalry | Competition, pricing pressure, new entrants |
| **TECHNOLOGICAL** | Innovation threats | Obsolescence, cybersecurity, disruption |
| **HUMAN_CAPITAL** | Workforce risks | Talent retention, labor disputes, skill gaps |
| **FINANCIAL** | Capital structure | Liquidity, debt, foreign exchange exposure |
| **REPUTATIONAL** | Brand and trust | PR crises, ESG controversies, trust erosion |
| **OTHER** | Miscellaneous | Company-specific edge cases |

---

## Quick Start: Analyst

You have a ticker and want a risk report in three commands.

**Step 1 — Install**
```bash
uv sync
```

**Step 2 — Download the 10-K filing(s)**
```bash
uv run sigmak download --ticker ABT
```

**Step 3 — Generate the year-over-year risk report**
```bash
uv run sigmak yoy --ticker ABT --years 2023 2024 2025
```

Output: `output/ABT_YoY_Risk_Analysis_2023_2025.md`

**Peer comparison** (optional):
```bash
uv run sigmak peers --ticker ABT --year 2025
```

Output: `output/ABT_Peer_Comparison_2025.md`

---

## Quick Start: Engineer

Setting up the full stack for the first time.

**Prerequisites**
- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- WSL2 if on Windows (required for ChromaDB persistence)

**Step 1 — Install dependencies**
```bash
uv sync
```

**Step 2 — Initialize the vector database**
```bash
uv run init_vector_db.py
```

**Step 3 — Configure API keys and settings**

```bash
cp api_keys.example.json api_keys.json
# Add your GOOGLE_API_KEY for LLM classification
```

Edit `config.yaml` for your environment (database paths, embedding model, drift thresholds). Full settings reference: [documentation/ARCHITECTURE.md](documentation/ARCHITECTURE.md#configuration).

**Run the full pipeline on a single filing**:
```bash
uv run sigmak analyze \
  --ticker TSLA \
  --year 2024 \
  --html-path data/filings/TSLA/2024/tsla-20241231.htm \
  --use-llm
```

Output: `output/results_TSLA_2024.json`

---

## CLI Reference

```
uv run sigmak <subcommand> [flags]
```

| Subcommand | What it does |
|---|---|
| `yoy` | Year-over-year risk analysis report for a ticker across multiple filing years |
| `peers` | Peer comparison report: scores a ticker against SIC-matched peer companies |
| `download` | Download 10-K HTM filings from SEC EDGAR for a ticker (and its peers) |
| `analyze` | Analyze a single filing: index, score, and export to JSON |
| `backfill` | Backfill existing LLM results from `output/results_*.json` into the vector DB |
| `peer-marketcap` | Update market-cap data for one or more peers in the SQLite DB |
| `inspect` | Inspect the SQLite and ChromaDB databases |
| `render` | Render a Markdown report to PDF via WeasyPrint |

**Common flags** (most subcommands):

| Flag | Description |
|---|---|
| `--ticker TICKER` | Target company ticker symbol (required for yoy, peers, download, analyze) |
| `--year` / `--years` | Filing year(s) to target |
| `--use-llm` | Enable Gemini LLM classification (requires `GOOGLE_API_KEY`) |
| `--db-only` | Use ChromaDB vector search only; no LLM calls |
| `--output-dir PATH` | Output directory (default: `./output`) |
| `--db-path PATH` | SQLite database path (default: `./database/sec_filings.db`) |

Full per-subcommand flag reference: `uv run sigmak <subcommand> --help`

---

## Human in the Loop — Alpha Phase

SigmaK is in active development and calibration. The proprietary severity and novelty scores are trained against a growing corpus of real 10-K filings, and the classification taxonomy is being refined through analyst feedback.

**What this means for users:**
- Scores are directionally reliable but should be reviewed before use in financial models
- Classification rationales and source citations are always included so reviewers can validate outputs
- Feedback on miscategorised risks or unexpected scores is actively used to improve the system

**To request a sample report, please contact peter.j.martinson@gmail.com**

If you\'re using SigmaK and want to contribute feedback or participate in the alpha testing program, reach out via the repository issues.

---

## Testing

```bash
# Run all tests
uv run python -m pytest

# Verbose output
uv run python -m pytest -v

# Type checking
uv run mypy .
```

The project follows strict **Test-Driven Development** (tests written before implementation) and **Single Responsibility** test design (one behaviour per test function).

For architecture details, deployment, REST API reference, drift detection, and programmatic usage, see [documentation/ARCHITECTURE.md](documentation/ARCHITECTURE.md).
