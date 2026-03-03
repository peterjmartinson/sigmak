**Title:** `Add CLI entry point scaffold: __main__.py + project.scripts`

**Body:**

## Overview

This issue wires up `uv run sigmak` as a real installed command. No business logic 
changes. The goal is a walking skeleton: the command runs, prints help, and exits 
cleanly. Every subsequent issue builds on this.

---

## Acceptance Criteria

- [ ] `pyproject.toml` contains a `[project.scripts]` entry:
      `sigmak = "sigmak.__main__:main"`
- [ ] `src/sigmak/cli/__init__.py` exists (empty)
- [ ] `src/sigmak/__main__.py` exists and implements a `build_parser()` + `main()` 
      function using `argparse`
- [ ] `--ticker TICKER` is a **global required flag** (before the subcommand)
- [ ] `--use-llm` and `--db-only` are **global mutually exclusive flags**
- [ ] Five subcommands are registered (none implemented yet — each prints 
      "not yet implemented" and exits 0):
      - `yoy`
      - `peers`
      - `download`
      - `inspect`
      - `render`
- [ ] If no subcommand is given: print help and exit 0 (no interactive loop)
- [ ] `uv run sigmak --ticker AAPL` prints help and exits 0
- [ ] `uv run sigmak --ticker AAPL yoy` prints "not yet implemented" and exits 0

---

## Tests (write these FIRST, before implementation)

File: `tests/test_main.py`

Write one test per behavior (SRP):
- `test_build_parser_requires_ticker` — calling `parse_args([])` raises SystemExit
- `test_build_parser_accepts_ticker` — `parse_args(['--ticker', 'AAPL'])` sets 
  `args.ticker == 'AAPL'`
- `test_build_parser_use_llm_flag` — `--use-llm` sets `args.use_llm == True`
- `test_build_parser_db_only_flag` — `--db-only` sets `args.db_only == True`
- `test_build_parser_flags_are_mutually_exclusive` — passing both `--use-llm` and 
  `--db-only` raises SystemExit
- `test_no_subcommand_exits_zero` — calling `main(['--ticker', 'AAPL'])` exits 0
- `test_yoy_subcommand_registered` — `parse_args(['--ticker', 'AAPL', 'yoy'])` 
  sets `args.command == 'yoy'`
- `test_peers_subcommand_registered` — same pattern for `peers`
- `test_download_subcommand_registered` — same pattern for `download`
- `test_inspect_subcommand_registered` — same pattern for `inspect`
- `test_render_subcommand_registered` — same pattern for `render`

All tests must pass before the PR is merged.

---

## Implementation Notes

- `build_parser()` must be a standalone function (makes it easy to test without 
  running main)
- `main()` should accept an optional `argv: list[str] | None = None` parameter 
  and pass it to `parser.parse_args()` — this makes testing without patching 
  sys.argv possible
- Subcommand dispatch in `main()` should use lazy imports 
  (e.g., `from sigmak.cli.yoy import run`) so missing implementations don't crash 
  on import — each CLI module can just have a stub `run()` that prints and returns
- Do NOT move or change any existing code in `scripts/` or `src/sigmak/`

---

## Files to Create / Modify

- `src/sigmak/__main__.py` (new)
- `src/sigmak/cli/__init__.py` (new, empty)
- `src/sigmak/cli/yoy.py` (new, stub only: `def run(**kwargs): print("not yet implemented")`)
- `src/sigmak/cli/peers.py` (new, stub)
- `src/sigmak/cli/download.py` (new, stub)
- `src/sigmak/cli/inspect_db.py` (new, stub)
- `src/sigmak/cli/render.py` (new, stub)
- `pyproject.toml` (add `[project.scripts]`)
- `tests/test_main.py` (new)

---

## Journal Entry

add to JOURNAL.md when complete

---

## README Update

Add a "Usage" section near the top of README.md documenting the new CLI:

```
## CLI Usage

    uv run sigmak --ticker AAPL yoy --years 2023 2024 2025
    uv run sigmak --ticker AAPL peers --year 2024
    uv run sigmak --ticker AAPL download
    uv run sigmak --ticker AAPL inspect
    uv run sigmak --ticker AAPL render --input output/AAPL_YoY.md

Global flags:
  --ticker TICKER   Target company (required)
  --use-llm         Use LLM for classification (requires GOOGLE_API_KEY)
  --db-only         Use ChromaDB only, no LLM calls
```