Add optional yfinance-based peer discovery adapter for fast demo peer lists

Goal
- Provide an optional, pluggable adapter that uses yfinance to produce a believable peer list for a single input ticker. This is intended for demos / proof-of-concept usage: fast, automatable, pragmatic, and easy to tune. It is explicitly not intended to replace the SEC-first canonical peer discovery; that remains the authoritative path for production workflows.

Non-goal
- Do not replace the SEC-based discovery in the repo. This feature is opt-in and demo-focused only.

Acceptance criteria (all must be satisfied)
1. Public behavior:
   - Given a ticker symbol input, a new method returns up to N peers (configurable; default N=10) with at least these fields: ticker, company_name, market_cap (nullable), exchange, source="yfinance".
2. Efficiency & safety:
   - Use bulk yfinance queries (yf.Tickers) rather than per-ticker loops.
3. Caching:
   - Implement local caching of raw yfinance enrichment payloads with TTL (configurable; default TTL = 24h) under PeerDiscoveryService cache_dir (or under cache_dir/peer_discovery/yfinance).
4. Rate limiting & backoff:
   - Enforce a configurable soft rate limit and exponential backoff with max retries (defaults provided). Failures should not crash the pipeline: surface clear errors and fall back to SEC-universe (if available) or return an empty list.
5. Filtering & heuristics:
   - Provide a pluggable filter pipeline with these defaults:
     - Primary match: match on Yahoo industry/sector when available, else best string match on company description keywords or exchange.
     - Believability default: include peers with market_cap >= max(min_absolute_cap, fraction * target_market_cap), with defaults min_absolute_cap = $50_000_000, fraction = 0.10.
     - If the target market_cap is missing, fallback to percentile-based selection among candidates.
     - If filtering yields fewer than min_peers (default 5), progressively relax fraction thresholds (0.10 -> 0.05 -> 0.02).
   - Deterministic ordering: returned list must be stable across identical inputs (sort by market_cap desc then ticker).
6. Tests:
   - Unit tests exist for: bulk fetch + cache hit, missing market_cap handling, filtering behavior, rate-limiting/backoff flow (network mocked), and min_peers relaxation.
7. Docs:
   - README updated with configuration options, opt-in instructions, data-source licensing/ToS note, and sample usage.
8. Opt-in:
   - The adapter must be disabled by default and enabled via config/env. No implicit behavioral change for existing callers.

Proposed configuration knobs (defaults shown)
- SIGMAK_PEER_YFINANCE_ENABLED = false
- SIGMAK_PEER_YFINANCE_N_PEERS = 10
- SIGMAK_PEER_YFINANCE_MIN_PEERS = 5
- SIGMAK_PEER_YFINANCE_TTL_SECONDS = 86400  # 24h
- SIGMAK_PEER_YFINANCE_MIN_FRACTION = 0.10
- SIGMAK_PEER_YFINANCE_MIN_ABS_CAP = 50_000_000
- SIGMAK_PEER_YFINANCE_RATE_LIMIT_RPS = 1
- SIGMAK_PEER_YFINANCE_MAX_RETRIES = 3
- SIGMAK_PEER_YFINANCE_BACKOFF_BASE_SECONDS = 0.5

High-level implementation plan (step-by-step)
1. Add adapter module
   - New file: src/sigmak/adapters/yfinance_adapter.py
   - Public API:
     - class YFinanceAdapter(cache_dir: Path, ttl_seconds: int, rate_limit_rps: float, max_retries: int, backoff_base: float)
     - method fetch_bulk(tickers: List[str]) -> Dict[str, Dict]  # normalized metadata per ticker (includes None for missing fields)
   - Responsibilities:
     - Use yf.Tickers for bulk requests.
     - Cache raw payloads to JSON files under cache_dir/peer_discovery/yfinance/{hash_of_sorted_tickers}.json with timestamp.
     - Normalize fields: ticker, shortName/companyName, marketCap (int or None), exchange, industry/sector, raw_source (timestamp, raw payload).
     - Respect rate limit and implement exponential backoff for transient failures (with retries).
     - Provide telemetry/logging on cache hit/miss, retries, and errors.

2. Add PeerDiscoveryService wrapper method (opt-in)
   - Modify src/sigmak/peer_discovery.py to add (non-breaking) method:
     - get_peers_via_yfinance(ticker: str, n: int = None, config: Optional[dict] = None) -> List[PeerRecord]
   - Behavior:
     - If env/config SIGMAK_PEER_YFINANCE_ENABLED is false: raise a clear NotEnabled error or return None/empty (prefer returning empty and logging).
     - Use existing PeerDiscoveryService.cache_dir and user_agent for cache location and polite logging.
     - Steps:
       1. Fetch target ticker metadata via yfinance adapter (single bulk call can include target).
       2. Build candidate universe via adapter heuristics: e.g., industry top companies if yfinance exposes, or search for related tickers (adapter returns top companies or top N tickers in industry when available).
       3. Enrich candidate universe via fetch_bulk.
       4. Apply believability filter pipeline described above.
       5. Return up to N peers with canonical fields and source="yfinance".

3. Tests (TDD-first)
   - New tests: tests/test_peer_discovery_yfinance.py
   - Use mocking (pytest + monkeypatch) for yfinance objects and for the filesystem cache. Alternatively use lightweight VCR fixtures for one integration-style test.
   - Test cases to include:
     - test_bulk_fetch_uses_single_yf_call_and_cache: assert adapter calls yf.Tickers once for multiple tickers and next call reads from cache.
     - test_filter_by_marketcap_and_fraction: given sample marketCaps, assert correct filtering sets.
     - test_missing_target_marketcap_percentile_fallback: if target market cap is None, ensure selection by percentile works.
     - test_min_peers_relaxation_progresses_correctly: if too few peers, thresholds relax deterministically.
     - test_backoff_and_retries_on_transient_errors: simulate transient failures and assert retries and fallback.
   - Keep tests deterministic and fast by mocking yfinance and time.sleep backoff.

4. Docs and UX
   - Update README.md section "Peer discovery" with:
     - Explanation of opt-in yfinance adapter, defaults, and configuration.
     - Example CLI/env to enable it.
     - ToS/legal note: explain yfinance is an unofficial Yahoo wrapper; consult Yahoo terms and make feature opt-in for client demos.
   - Add a JOURNAL.md entry describing why this demo path exists and tradeoffs.
   - Add whether the adapter is allowed to log raw yfinance payloads and where (sensitive info consideration).

5. CI
   - Ensure new tests run under CI and do not perform real network calls (mocking required).
   - Add any test fixtures to tests/fixtures and keep them small.

Files to add/modify (suggested)
- Add: src/sigmak/adapters/yfinance_adapter.py
- Modify: src/sigmak/peer_discovery.py (add wrapper method only; keep existing code unchanged otherwise)
- Add: tests/test_peer_discovery_yfinance.py
- Modify: README.md (Peer discovery section)
- Modify or add: JOURNAL.md (note about feature and tradeoffs)
- Add: .github/labels or project/meta as appropriate (optional)

Sample output (one example JSON object per peer)
[
  {
    "ticker": "APH",
    "company_name": "Amphenol Corporation",
    "market_cap": 35000000000,
    "exchange": "NYSE",
    "industry": "Electrical Products",
    "source": "yfinance",
    "enriched_at": "2026-02-20T12:34:56Z"
  },
  {
    "ticker": "TE",
    "company_name": "TE Connectivity Ltd.",
    "market_cap": 27000000000,
    "exchange": "NYSE",
    "industry": "Electrical Products",
    "source": "yfinance",
    "enriched_at": "2026-02-20T12:34:56Z"
  }
]

Security & compliance notes
- yfinance uses unofficial Yahoo endpoints. Document ToS implications in README and make the feature opt-in. Avoid recommending heavy, sustained scraping in client demos.
- Use a clear User-Agent on any HTTP wrappers (PeerDiscoveryService already uses user_agent for SEC calls; apply same pattern).
- Do not commit raw cached payloads to the repo.

Suggested tests / QA checklist (for you while developing locally)
- Unit tests pass with mocks only (no network).
- Integration test with one recorded fixture (VCR) that demonstrates end-to-end enrichment and filtering.
- Manual demo: enable adapter locally, run get_peers_via_yfinance for a few tickers, verify performance and believable output.
- Observe cache TTL behavior by altering system clock or TTL and asserting refresh occurs.

Suggested labels / assignees / milestone
- labels: enhancement, feature/demo, needs-tests, opt-in
- assignee: (leave blank or add yourself)
- milestone: v0.1-demo or similar

Questions / configuration items for you to decide (answer in comments on the Issue)
1. Default N peers for demos (I propose N=10; min_peers=5).
2. Should the adapter be disabled by default? (I recommend disabled by default; opt-in.)
3. Preferred cache location: default to existing PeerDiscoveryService.cache_dir/peer_discovery/yfinance? (recommended: yes)
4. Any legal constraint about mentioning Yahoo publicly in client-facing demo docs?

Checklist for the Issue body (to paste at the bottom of the Issue)
- [ ] Create adapter file src/sigmak/adapters/yfinance_adapter.py
- [ ] Add PeerDiscoveryService.get_peers_via_yfinance wrapper method
- [ ] Implement caching per TTL in cache_dir
- [ ] Implement rate-limiting and exponential backoff
- [ ] Implement believability filter pipeline and deterministic ordering
- [ ] Add unit tests and fixtures in tests/test_peer_discovery_yfinance.py
- [ ] Add README.md docs + JOURNAL.md entry
- [ ] Ensure CI runs tests with mocks (no external network)
- [ ] Manual demo verification and cache TTL test

---

If you want, I can also:
- Produce a compact checklist-only Issue body (shorter) for quick pasting.
- Draft the exact README paragraph and the minimal test skeleton (no production code) you can paste into your local environment to begin TDD.

---
<!-- AGENT_IMPLEMENTATION_PLAN_START -->
## Implementation Plan (ready to execute — do not edit this section manually)

> **Status**: APPROVED, NOT YET STARTED  
> **Agreed**: 2026-02-20  
> **Run tests with**: `uv run pytest`

### Locked-in decisions

| # | Decision |
|---|---|
| Candidate pool | `yf.Ticker(target).info["industryKey"]` → `yf.Industry(key)` probed defensively for top-companies list |
| DB write | Upsert `market_cap`, `exchange`, `industry` back into existing `peers` table via `upsert_peer()` as a side-effect |
| `PeerRecord` | `@dataclass` (fields: `ticker`, `company_name`, `market_cap: Optional[int]`, `exchange`, `industry`, `sector`, `source`, `enriched_at`) |
| Tests | All mocked with `monkeypatch`; no VCR, no real network |
| `yf.Industry` probe | Defensive: try `.top_companies` (DataFrame index), then `.tickers`, then `.members`; log which succeeded; gracefully return `[]` on total failure |

### Files to create / modify

| Action | File |
|---|---|
| CREATE | `src/sigmak/adapters/__init__.py` (empty package marker) |
| CREATE | `src/sigmak/adapters/yfinance_adapter.py` (adapter + `PeerRecord`) |
| MODIFY | `src/sigmak/peer_discovery.py` (add `get_peers_via_yfinance` wrapper method only; no other changes) |
| CREATE | `tests/test_peer_discovery_yfinance.py` (TDD-first, all mocked) |
| MODIFY | `README.md` (add yfinance adapter section) |
| MODIFY | `JOURNAL.md` (add feature entry) |

### Implementation steps (ordered)

1. **Create adapter package** — `src/sigmak/adapters/__init__.py` (empty)

2. **Write tests first** — `tests/test_peer_discovery_yfinance.py`:
   - `test_bulk_fetch_uses_single_yf_tickers_call_and_caches`
   - `test_filter_by_marketcap_and_fraction`
   - `test_missing_target_marketcap_percentile_fallback`
   - `test_min_peers_relaxation_progresses_correctly`
   - `test_backoff_and_retries_on_transient_errors`
   - `test_industry_object_probed_defensively`
   - `test_adapter_disabled_by_default`

3. **Implement `PeerRecord` dataclass and `YFinanceAdapter` class** — `src/sigmak/adapters/yfinance_adapter.py`:
   - `__init__(cache_dir, ttl_seconds, rate_limit_rps, max_retries, backoff_base)` — reads env vars for defaults
   - `_resolve_industry_key(ticker) -> Optional[str]`
   - `_get_industry_candidates(industry_key) -> List[str]` — defensive probe of `yf.Industry`
   - `fetch_bulk(tickers) -> Dict[str, Dict]` — `yf.Tickers`, normalize, cache to `cache_dir/yfinance/{hash}.json` with TTL
   - `_apply_filter_pipeline(candidates, target_market_cap) -> List[PeerRecord]` — fraction filter 0.10→0.05→0.02 relaxation; sort by `market_cap DESC NULLS LAST, ticker ASC`
   - `get_peers(ticker, n) -> List[PeerRecord]` — orchestrates above + calls `upsert_peer()` for each result

4. **Add wrapper to `PeerDiscoveryService`** — `src/sigmak/peer_discovery.py`:
   - `get_peers_via_yfinance(ticker: str, n: Optional[int] = None) -> List` 
   - Guards on `SIGMAK_PEER_YFINANCE_ENABLED`; instantiates `YFinanceAdapter` using `self.cache_dir` and `self.db_path`

5. **Update `README.md`** — "Peer discovery: yfinance adapter" section with opt-in config, knobs table, ToS note, sample snippet

6. **Update `JOURNAL.md`** — one entry for the feature and tradeoff rationale

### Config knobs (env vars)

```
SIGMAK_PEER_YFINANCE_ENABLED        = false      # must be set to "true" to enable
SIGMAK_PEER_YFINANCE_N_PEERS        = 10
SIGMAK_PEER_YFINANCE_MIN_PEERS      = 5
SIGMAK_PEER_YFINANCE_TTL_SECONDS    = 86400      # 24h
SIGMAK_PEER_YFINANCE_MIN_FRACTION   = 0.10
SIGMAK_PEER_YFINANCE_MIN_ABS_CAP    = 50000000
SIGMAK_PEER_YFINANCE_RATE_LIMIT_RPS = 1
SIGMAK_PEER_YFINANCE_MAX_RETRIES    = 3
SIGMAK_PEER_YFINANCE_BACKOFF_BASE   = 0.5
```

### Verification commands

```bash
# All unit tests pass with no network calls
uv run pytest tests/test_peer_discovery_yfinance.py -v

# Manual smoke-test (opt-in enabled)
SIGMAK_PEER_YFINANCE_ENABLED=true uv run python -c "
from sigmak.peer_discovery import PeerDiscoveryService
svc = PeerDiscoveryService()
print(svc.get_peers_via_yfinance('MSFT', n=5))
"

# Confirm upsert side-effect
sqlite3 database/sec_filings.db "SELECT ticker, market_cap, industry FROM peers LIMIT 10;"
```
<!-- AGENT_IMPLEMENTATION_PLAN_END -->
