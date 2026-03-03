**Relates to:** #103, #106

## Overview

By default, `uv run sigmak --ticker AAPL peers --year 2024` should use yfinance-based peer selection (via `PeerDiscoveryService.get_peers_via_yfinance`). The old SIC/EDGAR-based selection should only be used when explicitly requested via a new `--sic-only` flag. This will allow demo_peer_discovery.py and related legacy scripts to be safely deleted with no loss of yfinance peer selection functionality.

## Acceptance Criteria

- [ ] At CLI startup, always set `SIGMAK_PEER_YFINANCE_ENABLED=true` unless it is already set in the environment.
- [ ] Add `--sic-only` boolean flag (`action="store_true"`) to the `peers` subparser in `__main__.py`.
- [ ] Propagate `use_sic_only` all the way to `run_peer_comparison()` in `reports/peer_report.py`.
- [ ] In `run_peer_comparison()`, use yfinance-based peer selection by default (extracting `.ticker` from returned `PeerRecord` list). If `use_sic_only` is true, use SIC/EDGAR selection logic (`find_peers_for_ticker`).
- [ ] CLI help/README updated to clearly document this behavior (yfinance default, SIC is opt-out).
- [ ] All existing and new tests reflect this: default is yfinance, flag can switch to SIC logic.
- [ ] Example: `uv run sigmak --ticker AAPL peers --year 2024` uses yfinance; `uv run sigmak --ticker AAPL peers --year 2024 --sic-only` uses SEC/EDGAR fallback

## Tests

- `test_peers_sic_only_flag_registered` — parses correctly, defaults to False, True when present
- `test_run_peer_comparison_sic_only_calls_find_peers_for_ticker`
- `test_run_peer_comparison_defaults_to_yfinance`
- End-to-end and integration CLI tests

## Files to Modify

- `src/sigmak/__main__.py`
- `src/sigmak/cli/peers.py`
- `src/sigmak/reports/peer_report.py`
- CLI test suite, README, and usage docs

---

Once done, `scripts/demo_peer_discovery.py` and any README references to it can be safely removed.