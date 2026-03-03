**Title:** `CLI: implement yoy subcommand by delegating to existing script logic`

**Body:**

## Overview

Wire `uv run sigmak --ticker AAPL yoy --years 2023 2024 2025` to run the full 
year-over-year analysis. The business logic already exists in 
`scripts/generate_yoy_report.py`. The task is to move that logic into 
`src/sigmak/reports/yoy_report.py` (library layer) and have 
`src/sigmak/cli/yoy.py` (CLI layer) call it.

Do NOT delete `scripts/generate_yoy_report.py` yet — it stays as-is until 
Issue 6 (cleanup). Do NOT break any existing tests.

---

## Acceptance Criteria

- [ ] `src/sigmak/reports/__init__.py` exists
- [ ] `src/sigmak/reports/yoy_report.py` contains the business logic extracted 
      from `scripts/generate_yoy_report.py`:
      - `validate_cached_result()`
      - `enrich_result_with_classification()`
      - `load_or_analyze_filing()`
      - `calculate_risk_similarity()`
      - `identify_risk_changes()`
      - `calculate_category_distribution()`
      - `extract_category_from_text()`
      - `suggest_categories_from_keywords()`
      - `load_filing_provenance()`
      - `is_valid_risk_paragraph()`
      - `generate_markdown_report()`
      - A new `run_yoy_analysis()` function that accepts `ticker`, `years`, 
        `use_llm`, `db_only` and orchestrates the full flow (the body of 
        `main()` from the script)
- [ ] `src/sigmak/cli/yoy.py` replaces its stub with:
      `def run(ticker, years, use_llm, db_only): ...`
      which calls `run_yoy_analysis()` from `sigmak.reports.yoy_report`
- [ ] `__main__.py` passes `--years` list, `--use-llm`, and `--db-only` to 
      `cli.yoy.run()`
- [ ] `--years` argument on `yoy` subcommand: `nargs="+"`, `type=int`, 
      default `[2023, 2024, 2025]`
- [ ] The script `scripts/generate_yoy_report.py` is updated to import from 
      `sigmak.reports.yoy_report` instead of duplicating logic (i.e., its 
      `main()` now calls `run_yoy_analysis()`)
- [ ] All existing tests still pass
- [ ] `uv run sigmak --ticker HURC yoy --years 2023 2024 2025` produces the 
      same output file as before

---

## Tests (write FIRST)

File: `tests/test_cli_yoy.py`

- `test_yoy_run_calls_run_yoy_analysis` — mock `sigmak.reports.yoy_report.run_yoy_analysis`; 
  call `cli.yoy.run(ticker='AAPL', years=[2023,2024], use_llm=False, db_only=False)`; 
  assert the mock was called with correct args
- `test_yoy_main_dispatch_calls_cli_run` — mock `sigmak.cli.yoy.run`; call 
  `main(['--ticker', 'AAPL', 'yoy', '--years', '2023', '2024'])`; assert mock called 
  with `ticker='AAPL'`
- `test_yoy_default_years` — mock `sigmak.cli.yoy.run`; call 
  `main(['--ticker', 'AAPL', 'yoy'])`; assert `years=[2023, 2024, 2025]`
- `test_yoy_db_only_flag_passed` — mock `sigmak.cli.yoy.run`; verify `db_only=True` 
  is forwarded when `--db-only` is given

---

## Implementation Notes

- The refactor is a pure extract-and-delegate: copy functions from the script 
  to the library module, update imports, do not change function signatures or logic
- `scripts/generate_yoy_report.py` becomes a thin wrapper:
  its `main()` just calls `from sigmak.reports.yoy_report import run_yoy_analysis`
  and passes its parsed args through
- Type hints and mypy compliance must be maintained throughout

---

## Files to Create / Modify

- `src/sigmak/reports/__init__.py` (new, empty)
- `src/sigmak/reports/yoy_report.py` (new — extracted logic)
- `src/sigmak/cli/yoy.py` (update stub to real implementation)
- `scripts/generate_yoy_report.py` (update `main()` to delegate)
- `tests/test_cli_yoy.py` (new)

---

## Journal Entry (add to JOURNAL.md when complete)
