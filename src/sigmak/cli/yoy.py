"""CLI handler for the `yoy` subcommand."""


def run(
    ticker: str,
    years: list[int],
    use_llm: bool,
    db_only: bool,
    **_: object,
) -> None:
    """Delegate to the yoy report library.

    Args:
        ticker: Target company ticker symbol.
        years: Filing years to analyse.
        use_llm: Force LLM-enriched classification (bypasses cache).
        db_only: Use ChromaDB only; no LLM calls.
        **_: Extra keys from argparse Namespace (e.g. ``command``) are ignored.
    """
    from sigmak.reports.yoy_report import run_yoy_analysis

    run_yoy_analysis(ticker=ticker, years=years, use_llm=use_llm, db_only=db_only)
