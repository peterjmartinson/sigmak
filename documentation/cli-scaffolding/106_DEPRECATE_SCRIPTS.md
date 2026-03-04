**Title:** `Deprecate /scripts: convert to thin wrappers or remove`

**Body:**

## Overview

Now that all business logic has been migrated to `src/sigmak/` and all five CLI 
subcommands are wired, the files in `/scripts` are redundant. This issue converts 
them to minimal, clearly-marked deprecated wrappers (or removes them if they add 
no value as compatibility shims).

---

## Acceptance Criteria

### For each script, choose one:

**Option A — Keep as compatibility shim** (if the script has a meaningfully 
different interface or is referenced in README):
- Reduce the script to ~10 lines: argument parsing + call to the new CLI
- Add a visible deprecation notice at the top: 
  `# DEPRECATED: use 'uv run sigmak ...' instead`

**Option B — Delete** (if the script is purely internal / dev tooling):
- Remove the file
- Confirm no existing tests import from it directly

### Specific recommendations:
- `generate_yoy_report.py` → Option A (documented in README)
- `generate_peer_comparison_report.py` → Option A (documented in README)
- `download_peers_and_target.py` → Option A
- `analyze_filing.py` → Option B (superseded by `yoy` subcommand)
- `batch_analyze_tesla.py` → Option B (dev/test artifact, company-specific)
- `demo_peer_discovery.py` → Option B (dev demo, not user-facing)
- `inspect_chroma.py` → Option A (useful standalone debug tool)
- `md_to_pdf.py` → Option A (some users may call directly)
- `backfill_llm_cache_to_chroma.py` → Option A (operational maintenance tool)
- `populate_peer_marketcap.py` → Option A (operational maintenance tool)

### Additional Criteria:
- [ ] All unit tests still pass after removals
- [ ] README updated to remove references to `python scripts/...` invocations and 
      replace with `uv run sigmak ...` equivalents
- [ ] README retains documentation of `backfill` and `populate_peer_marketcap` 
      as maintenance utilities (they have no CLI subcommand equivalent)
- [ ] `uv run sigmak --help` output is accurate and documented in README

---

## Tests (write FIRST)

File: `tests/test_scripts_deprecated.py`

For each script kept as a shim:
- `test_generate_yoy_report_is_importable` — `import scripts.generate_yoy_report` 
  does not raise (confirms shim is syntactically valid)
- Same pattern for each kept shim

For deleted scripts:
- Confirm no existing test file imports them (grep-level check, document findings 
  in PR description)

---

## Files to Create / Modify

- `scripts/generate_yoy_report.py` (reduce to shim or delete)
- `scripts/generate_peer_comparison_report.py` (reduce to shim or delete)
- `scripts/download_peers_and_target.py` (reduce to shim)
- `scripts/analyze_filing.py` (delete)
- `scripts/batch_analyze_tesla.py` (delete)
- `scripts/demo_peer_discovery.py` (delete)
- `scripts/inspect_chroma.py` (reduce to shim)
- `scripts/md_to_pdf.py` (reduce to shim)
- `scripts/backfill_llm_cache_to_chroma.py` (reduce to shim)
- `scripts/populate_peer_marketcap.py` (reduce to shim)
- `README.md` (update all `python scripts/...` references)

---

## Journal Entry (add to JOURNAL.md when complete)

---

## README Update (required)

Replace the "Quick Start" section's `python scripts/...` commands with 
`uv run sigmak ...` equivalents. Example:

Before:
```
python scripts/generate_yoy_report.py AAPL 2023 2024 2025
```

After:
```
uv run sigmak --ticker AAPL yoy --years 2023 2024 2025
```
