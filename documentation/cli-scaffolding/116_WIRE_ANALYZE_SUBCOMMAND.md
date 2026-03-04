**Title:** `CLI: wire analyze subcommand`

**Body:**

## Overview

Wire the core filing analysis pipeline into the `sigmak` CLI:

```
uv run sigmak analyze --ticker AAPL --year 2024 --html-path data/filings/AAPL/2024/aapl_2024_10k.htm
```

This is the critical missing step that parses a downloaded HTM filing, runs
`IntegrationPipeline` (chunk → embed → classify), and writes results to
ChromaDB and the output JSON cache.  Without this command, users cannot
populate the vector database from the CLI.

---

## Acceptance Criteria

- [ ] `src/sigmak/cli/analyze.py` created with:
      ```python
      def run(
          ticker: str,
          year: int,
          html_path: str,
          persist_path: str = "./database",
          output_dir: str = "./output",
          use_llm: bool = False,
          **_: object,
      ) -> None:
      ```
      which delegates to `IntegrationPipeline` (matching the logic in
      `scripts/analyze_filing.py`).
- [ ] `__main__.py` gains an `analyze` subcommand with args:
      - `--ticker TICKER` (required)
      - `--year INT` (required)
      - `--html-path PATH` (required) — path to the downloaded HTM filing
      - `--persist-path PATH` (default `./database`)
      - `--output-dir PATH` (default `./output`)
      - `--use-llm` / `--db-only` (mutually exclusive, same pattern as `yoy`)
- [ ] `scripts/analyze_filing.py` `main()` delegates to `cli.analyze.run()`
- [ ] Output JSON written to `output/results_{TICKER}_{YEAR}.json` (preserves
      existing script behaviour)
- [ ] Exits with informative error if the HTM file does not exist
- [ ] All existing tests still pass

---

## Tests (write FIRST)

File: `tests/test_cli_analyze.py`
- `test_analyze_main_dispatch_calls_cli_run` — mock `sigmak.cli.analyze.run`;
  call `main(['analyze', '--ticker', 'AAPL', '--year', '2024', '--html-path',
  'x.htm'])`; assert mock called with `ticker='AAPL'`, `year=2024`,
  `html_path='x.htm'`
- `test_analyze_missing_html_exits_gracefully` — call `cli.analyze.run()` with
  a path that does not exist; assert `SystemExit(1)`
- `test_analyze_persist_path_forwarded` — assert `--persist-path ./mydb`
  forwarded correctly

---

## Files to Create / Modify

- `src/sigmak/cli/analyze.py` (new)
- `src/sigmak/__main__.py` (add `analyze` subcommand)
- `scripts/analyze_filing.py` (update `main()` to delegate)
- `tests/test_cli_analyze.py` (new)

---

## Journal Entry (add to JOURNAL.md when complete)
