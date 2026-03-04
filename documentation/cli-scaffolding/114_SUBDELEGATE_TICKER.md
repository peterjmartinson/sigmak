Currently, the `uv run sigmak` command requires a `--ticker` parameter, even for subcommands where a ticker is not relevant. For example, `uv run sigmak inspect` still requires `--ticker`, even though the inspect subcommand does not use this value. Conversely, commands like `uv run sigmak --ticker AAPL yoy` do need the ticker.

**Expected behavior:**
- Only require the `--ticker` parameter for subcommands that actually use it (e.g., `yoy`).
- Subcommands such as `inspect` should *not* require a `--ticker` parameter.

**Suggested changes:**
- Refactor the CLI parameter handling so that the necessity and enforcement of `--ticker` are delegated to individual subcommands.
- Update documentation and help output to clarify for each subcommand whether `--ticker` is required.

**Example Scenarios:**
- `uv run sigmak --ticker AAPL yoy` ▶️ should require `--ticker` and work as before.
- `uv run sigmak inspect` ▶️ should NOT require `--ticker`.

This will improve usability and reduce confusion for end users running subcommands where `--ticker` is irrelevant.
