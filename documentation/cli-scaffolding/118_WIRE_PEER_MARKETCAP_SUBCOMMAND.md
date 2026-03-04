**Title:** `CLI: wire peer-marketcap subcommand`

**Body:**

## Overview

Wire the peer market-cap population utility into the `sigmak` CLI:

```
uv run sigmak peer-marketcap --ticker AAPL
uv run sigmak peer-marketcap --all
```

This command uses `yfinance` to fetch current market capitalisation data for
peers already stored in the SQLite filings database and writes it back.
Market-cap data feeds the peer weighting logic in comparison reports.

---

## Acceptance Criteria

- [ ] `src/sigmak/cli/peer_marketcap.py` created with:
      ```python
      def run(
          tickers: list[str] | None = None,
          all_peers: bool = False,
          delay: float = 1.0,
          db_path: str = "./database/sec_filings.db",
          **_: object,
      ) -> None:
      ```
      which delegates to `sigmak.filings_db.populate_market_cap` (matching the
      logic in `scripts/populate_peer_marketcap.py`).
- [ ] `__main__.py` gains a `peer-marketcap` subcommand with args:
      - `--ticker TICKER [TICKER ...]` (one or more; mutually exclusive with
        `--all`)
      - `--all` (update every peer in the DB; mutually exclusive with
        `--ticker`)
      - `--delay FLOAT` (seconds between yfinance requests; default `1.0`)
      - `--db-path PATH` (default `./database/sec_filings.db`)
- [ ] `peer-marketcap` does **not** require a `--ticker` global flag (it
      accepts an optional local `--ticker` list instead)
- [ ] Exits with an informative error if neither `--ticker` nor `--all` is
      provided
- [ ] `scripts/populate_peer_marketcap.py` `main()` delegates to
      `cli.peer_marketcap.run()`
- [ ] All existing tests still pass

---

## Tests (write FIRST)

File: `tests/test_cli_peer_marketcap.py`
- `test_peer_marketcap_dispatch_ticker` — mock `sigmak.cli.peer_marketcap.run`;
  call `main(['peer-marketcap', '--ticker', 'AAPL', 'MSFT'])`; assert mock
  called with `tickers=['AAPL', 'MSFT']`
- `test_peer_marketcap_dispatch_all` — mock run; call
  `main(['peer-marketcap', '--all'])`; assert `all_peers=True`
- `test_peer_marketcap_delay_forwarded` — assert `--delay 0.5` forwarded as
  `delay=0.5`
- `test_peer_marketcap_no_ticker_required` — `main(['peer-marketcap', '--all'])`
  must not raise due to a missing global `--ticker`

---

## Files to Create / Modify

- `src/sigmak/cli/peer_marketcap.py` (new)
- `src/sigmak/__main__.py` (add `peer-marketcap` subcommand)
- `scripts/populate_peer_marketcap.py` (update `main()` to delegate)
- `tests/test_cli_peer_marketcap.py` (new)

---

## Journal Entry (add to JOURNAL.md when complete)
