**Title:** `CLI: implement download subcommand by delegating to existing script logic`

**Body:**

## Overview

Wire `uv run sigmak --ticker NVDA download --year 2024` to download 10-K filings 
from SEC EDGAR. The logic lives in `scripts/download_peers_and_target.py`. Move 
orchestration logic into `src/sigmak/cli/download.py`.

Note: unlike the report scripts, `download_peers_and_target.py` has relatively 
little business logic worth extracting to a library module — its `main()` is 
primarily coordination. The CLI module can contain the orchestration directly, 
delegating to the existing `TenKDownloader` and `PeerDiscoveryService` library 
classes as the script already does.

---

## Acceptance Criteria

- [ ] `src/sigmak/cli/download.py` replaces its stub with a real `run()` function:
      ```python
      def run(ticker: str, years: list[int] | None, include_peers: bool, 
              db_only: bool, db_path: str = "./database/sec_filings.db",
              download_dir: str = "./data/filings") -> None:
      ```
      This function contains the orchestration logic from `scripts/download_peers_and_target.py`'s `main()`
- [ ] `__main__.py` wires `download` subcommand args:
      - `--years INT [INT...]` (optional; if omitted, downloads latest available)
      - `--include-peers` (boolean flag, default False)
      - `--max-peers INT` (default 6)
- [ ] `scripts/download_peers_and_target.py` `main()` delegates to `cli.download.run()`
- [ ] All existing tests still pass
- [ ] `uv run sigmak --ticker NVDA download --year 2024` runs without error

---

## Tests (write FIRST)

File: `tests/test_cli_download.py`

- `test_download_run_signature_accepts_required_args` — call `run()` with all 
  required args mocked (mock TenKDownloader and PeerDiscoveryService); assert 
  no exception raised
- `test_download_main_dispatch_calls_cli_run` — mock `sigmak.cli.download.run`; 
  call `main(['--ticker', 'NVDA', 'download'])`; assert mock called with 
  `ticker='NVDA'`
- `test_download_include_peers_flag` — mock `sigmak.cli.download.run`; call with 
  `--include-peers`; assert `include_peers=True` forwarded
- `test_download_years_forwarded` — mock `sigmak.cli.download.run`; call with 
  `--years 2023 2024`; assert `years=[2023, 2024]` forwarded

---

## Files to Create / Modify

- `src/sigmak/cli/download.py` (update stub to real implementation)
- `scripts/download_peers_and_target.py` (update main() to delegate)
- `tests/test_cli_download.py` (new)

---

## Journal Entry (add to JOURNAL.md when complete)
