**Title:** `CLI: implement inspect and render subcommands`

**Body:**

## Overview

Wire the two remaining utility subcommands:
- `uv run sigmak --ticker AAPL inspect` — delegates to `scripts/inspect_chroma.py` logic
- `uv run sigmak --ticker AAPL render --input output/AAPL_YoY.md` — delegates to 
  `scripts/md_to_pdf.py` logic

Note: `render` does not conceptually need `--ticker`, but it stays consistent with 
the global required flag. The `--ticker` value can be ignored in the render 
implementation.

---

## Acceptance Criteria

### inspect
- [ ] `src/sigmak/cli/inspect_db.py` replaces stub with:
      ```python
      def run(ticker: str, chroma_dir: str = "./database", 
              max_sample: int = 5) -> None:
      ```
      which calls the inspection logic from `scripts/inspect_chroma.py`
- [ ] `__main__.py` `inspect` subcommand gains optional args:
      - `--chroma-dir PATH` (default `./database`)
      - `--max-sample INT` (default 5)
- [ ] `scripts/inspect_chroma.py` `main()` delegates to `cli.inspect_db.run()`

### render
- [ ] `src/sigmak/cli/render.py` replaces stub with:
      ```python
      def run(input_path: str, output_path: str | None = None, 
              css_path: str = "styles/report.css", 
              title: str | None = None) -> None:
      ```
      which calls `convert_md_to_pdf()` from `scripts/md_to_pdf.py` (or extract 
      that function to `src/sigmak/reports/pdf_renderer.py`)
- [ ] `__main__.py` `render` subcommand:
      - `--input PATH` (required)
      - `--output PATH` (optional)
      - `--css PATH` (default `styles/report.css`)
      - `--title STR` (optional)
- [ ] `scripts/md_to_pdf.py` `main()` delegates to `cli.render.run()`
- [ ] `render` gracefully exits with an informative error message if `weasyprint` 
      is not installed (it is an optional dependency)
- [ ] All existing tests still pass

---

## Tests (write FIRST)

File: `tests/test_cli_inspect.py`
- `test_inspect_main_dispatch_calls_cli_run` — mock `sigmak.cli.inspect_db.run`; 
  call `main(['--ticker', 'AAPL', 'inspect'])`; assert mock called with 
  `ticker='AAPL'`
- `test_inspect_chroma_dir_forwarded` — mock `sigmak.cli.inspect_db.run`; call 
  with `--chroma-dir ./mydb`; assert `chroma_dir='./mydb'` forwarded

File: `tests/test_cli_render.py`
- `test_render_main_dispatch_calls_cli_run` — mock `sigmak.cli.render.run`; call 
  `main(['--ticker', 'AAPL', 'render', '--input', 'output/test.md'])`; assert 
  mock called with `input_path='output/test.md'`
- `test_render_missing_weasyprint_exits_gracefully` — mock `weasyprint` as 
  unavailable (ImportError); call `cli.render.run(input_path='x.md')`; assert 
  SystemExit or a clean error message is printed (no unhandled traceback)
- `test_render_output_path_forwarded` — mock run internals; verify `--output` 
  arg forwarded correctly

---

## Files to Create / Modify

- `src/sigmak/cli/inspect_db.py` (update stub)
- `src/sigmak/cli/render.py` (update stub)
- `src/sigmak/reports/pdf_renderer.py` (optional extraction of `convert_md_to_pdf`)
- `scripts/inspect_chroma.py` (update main() to delegate)
- `scripts/md_to_pdf.py` (update main() to delegate)
- `tests/test_cli_inspect.py` (new)
- `tests/test_cli_render.py` (new)

---

## Journal Entry (add to JOURNAL.md when complete)
