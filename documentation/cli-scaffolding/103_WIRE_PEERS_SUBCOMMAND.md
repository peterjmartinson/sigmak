**Title:** `CLI: implement peers subcommand by delegating to existing script logic`

**Body:**

## Overview

Wire `uv run sigmak --ticker AAPL peers --year 2024` to run the full peer 
comparison report. The business logic lives in 
`scripts/generate_peer_comparison_report.py`. Move it into 
`src/sigmak/reports/peer_report.py` and have `src/sigmak/cli/peers.py` call it.

Do NOT delete `scripts/generate_peer_comparison_report.py` yet.

---

## Acceptance Criteria

- [ ] `src/sigmak/reports/peer_report.py` contains business logic extracted from 
      `scripts/generate_peer_comparison_report.py`:
      - `find_html_in_dir()`
      - `locate_filing_html()`
      - `ensure_filing()`
      - `validate_cached_result()`
      - `load_or_analyze_with_cache()`
      - `filter_boilerplate()`
      - `compute_severity_avg()`
      - `compute_category_distribution()`
      - `generate_markdown_report()`
      - A new `run_peer_comparison()` function that accepts `ticker`, `year`, 
        `max_peers`, `explicit_peers`, `db_only`, `db_path`, `download_dir` 
        and orchestrates the full flow
- [ ] `src/sigmak/cli/peers.py` replaces its stub with a `run()` that calls 
      `run_peer_comparison()` from `sigmak.reports.peer_report`
- [ ] `__main__.py` wires `peers` subcommand args:
      - `--year INT` (required)
      - `--max-peers INT` (default 6)
      - `--peers TICKER [TICKER...]` (optional)
- [ ] `scripts/generate_peer_comparison_report.py` main() delegates to 
      `run_peer_comparison()`
- [ ] All existing tests still pass
- [ ] `uv run sigmak --ticker NVDA peers --year 2024` runs without error

---

## Tests (write FIRST)

File: `tests/test_cli_peers.py`

- `test_peers_run_calls_run_peer_comparison` — mock `sigmak.reports.peer_report.run_peer_comparison`; 
  call `cli.peers.run(ticker='NVDA', year=2024, max_peers=6, explicit_peers=None, db_only=False)`;
  assert mock called with correct args
- `test_peers_main_dispatch_calls_cli_run` — mock `sigmak.cli.peers.run`; call 
  `main(['--ticker', 'NVDA', 'peers', '--year', '2024'])`; assert mock called with 
  `ticker='NVDA'`, `year=2024`
- `test_peers_explicit_peers_forwarded` — mock `sigmak.cli.peers.run`; call with 
  `--peers AAPL MSFT`; assert `explicit_peers=['AAPL', 'MSFT']` forwarded
- `test_peers_db_only_flag_forwarded` — verify `db_only=True` forwarded when 
  `--db-only` global flag given

---

## Files to Create / Modify

- `src/sigmak/reports/peer_report.py` (new — extracted logic)
- `src/sigmak/cli/peers.py` (update stub)
- `scripts/generate_peer_comparison_report.py` (delegate main())
- `tests/test_cli_peers.py` (new)

---

## Journal Entry (add to JOURNAL.md when complete)
