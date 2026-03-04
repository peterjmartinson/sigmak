**Title:** `CLI: wire backfill subcommand`

**Body:**

## Overview

Wire the LLM classification backfill utility into the `sigmak` CLI:

```
uv run sigmak backfill --write
uv run sigmak backfill --dry-run
```

This command reads existing `output/*.json` result files and ensures their
LLM classifications are properly persisted in both SQLite and ChromaDB.  It
is a recovery/maintenance tool — useful when the analysis pipeline ran without
a `GOOGLE_API_KEY` and classifications need to be populated retroactively.

---

## Acceptance Criteria

- [ ] `src/sigmak/cli/backfill.py` created with:
      ```python
      def run(
          write: bool = False,
          dry_run: bool = False,
          output_dir: str = "./output",
          db_path: str = "./database/sec_filings.db",
          **_: object,
      ) -> None:
      ```
      which delegates to the logic in `scripts/backfill_llm_cache_to_chroma.py`.
- [ ] `__main__.py` gains a `backfill` subcommand with args:
      - `--write` (store results; mutually exclusive with `--dry-run`)
      - `--dry-run` (preview only; default if neither flag given)
      - `--output-dir PATH` (default `./output`)
      - `--db-path PATH` (default `./database/sec_filings.db`)
- [ ] `backfill` does **not** require `--ticker` (operates on all output files)
- [ ] `scripts/backfill_llm_cache_to_chroma.py` `main()` delegates to
      `cli.backfill.run()`
- [ ] Dry-run mode prints a summary of what would be written without modifying
      any database (preserves existing script behaviour)
- [ ] All existing tests still pass

---

## Tests (write FIRST)

File: `tests/test_cli_backfill.py`
- `test_backfill_main_dispatch_calls_cli_run` — mock `sigmak.cli.backfill.run`;
  call `main(['backfill', '--dry-run'])`; assert mock called with
  `dry_run=True`
- `test_backfill_write_flag_forwarded` — assert `--write` forwarded as
  `write=True`
- `test_backfill_output_dir_forwarded` — assert `--output-dir ./custom`
  forwarded as `output_dir='./custom'`
- `test_backfill_no_ticker_required` — `main(['backfill', '--dry-run'])` must
  not raise `SystemExit` due to a missing `--ticker`

---

## Files to Create / Modify

- `src/sigmak/cli/backfill.py` (new)
- `src/sigmak/__main__.py` (add `backfill` subcommand)
- `scripts/backfill_llm_cache_to_chroma.py` (update `main()` to delegate)
- `tests/test_cli_backfill.py` (new)

---

## Journal Entry (add to JOURNAL.md when complete)
