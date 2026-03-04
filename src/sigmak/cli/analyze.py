"""CLI handler for the `analyze` subcommand.

Parses a downloaded HTM filing, runs IntegrationPipeline (chunk → embed →
classify), and writes results to ChromaDB and the output JSON cache.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict


def _classify_risks(
    risks: list[Dict[str, Any]],
    use_llm: bool,
    pipeline: Any,
) -> None:
    """Enrich each risk dict in-place with category / confidence / method.

    Uses the same precedence as the original script:
    1. Already classified in metadata → copy through
    2. ``use_llm=True`` or ``GOOGLE_API_KEY`` present → RiskClassificationService
    3. Fallback → vector-DB similarity search via drift_system
    """
    from sigmak.risk_classification_service import RiskClassificationService

    use_full_service = False
    classification_service = None
    if use_llm or os.getenv("GOOGLE_API_KEY"):
        classification_service = RiskClassificationService(
            drift_system=pipeline.drift_system
        )
        use_full_service = True

    for i, risk in enumerate(risks, 1):
        try:
            if risk.get("metadata", {}).get("category"):
                risk["category"] = risk["metadata"]["category"]
                risk["category_confidence"] = risk["metadata"].get("confidence", 0.0)
                risk["classification_method"] = risk["metadata"].get(
                    "classification_source", "vector"
                )
                continue

            if use_full_service and classification_service:
                llm_result, source = classification_service.classify_with_cache_first(
                    risk["text"]
                )
                risk["category"] = llm_result.category.value
                risk["category_confidence"] = llm_result.confidence
                risk["classification_method"] = source
                print(f"Risk #{i}: classified as {risk['category']} (source={source})")
            else:
                try:
                    embedding = (
                        pipeline.indexing_pipeline.embeddings.encode([risk["text"]])[0].tolist()
                    )
                    cache_results = pipeline.drift_system.similarity_search(
                        query_embedding=embedding, n_results=1
                    )
                    if cache_results and cache_results[0].get("similarity_score", 0.0) >= 0.8:
                        top = cache_results[0]
                        risk["category"] = top.get("category", "UNCATEGORIZED")
                        risk["category_confidence"] = float(top.get("confidence", 0.0))
                        risk["classification_method"] = "vector_db"
                        print(f"Risk #{i}: classified from vector DB as {risk['category']}")
                    else:
                        risk["category"] = "UNCATEGORIZED"
                        risk["category_confidence"] = 0.0
                        risk["classification_method"] = "db_only_no_match"
                        print(f"Risk #{i}: no cached classification found; marked UNCATEGORIZED")
                except Exception as exc:
                    print(f"Risk #{i}: cache-only classification failed: {exc}")
                    risk["category"] = "UNCATEGORIZED"
                    risk["category_confidence"] = 0.0
                    risk["classification_method"] = "error"
        except Exception as exc:
            print(f"Risk #{i}: classification failed: {exc}")


def run(
    ticker: str,
    year: int,
    html_path: str,
    persist_path: str = "./database",
    output_dir: str = "./output",
    use_llm: bool = False,
    db_only: bool = False,
    **_: object,
) -> None:
    """Analyze a downloaded SEC 10-K HTM filing and persist results.

    Parameters
    ----------
    ticker:
        Company ticker symbol (will be upper-cased).
    year:
        Filing year (e.g. 2024).
    html_path:
        Path to the downloaded HTM filing.
    persist_path:
        ChromaDB / SQLite persistence directory (default: ``./database``).
    output_dir:
        Directory for the output JSON cache (default: ``./output``).
    use_llm:
        Force LLM classification even without GOOGLE_API_KEY check.
    db_only:
        Skip LLM entirely; use vector-DB similarity only.
    """
    from sigmak.integration import IntegrationPipeline

    ticker = ticker.upper()
    html = Path(html_path)

    if not html.exists():
        print(f"Error: HTM file not found: {html}", file=sys.stderr)
        sys.exit(1)

    print(f"\nInitializing analysis pipeline...")
    pipeline = IntegrationPipeline(
        persist_path=persist_path,
        db_only_classification=db_only,
    )

    print(f"Analyzing {ticker} {year} from {html_path}...")
    result = pipeline.analyze_filing(
        html_path=html_path,
        ticker=ticker,
        filing_year=year,
        retrieve_top_k=5,
    )

    print(f"\nAnalysis complete.")
    print(f"  Chunks indexed : {result.metadata['chunks_indexed']}")
    print(f"  Risks analyzed : {result.metadata['risks_analyzed']}")
    print(f"  Total time     : {result.metadata['total_latency_ms']:.0f}ms")

    for i, risk in enumerate(result.risks, 1):
        print(f"\n--- Risk #{i} ---")
        print(f"Severity : {risk['severity']['value']:.2f} - {risk['severity']['explanation']}")
        print(f"Novelty  : {risk['novelty']['value']:.2f} - {risk['novelty']['explanation']}")
        print(f"Text     : {risk['text'][:200]}...")

    if not db_only:
        print("\nClassifying risks (vector DB with LLM fallback)...")
        _classify_risks(result.risks, use_llm=use_llm, pipeline=pipeline)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / f"results_{ticker}_{year}.json"
    output_file.write_text(result.to_json())
    print(f"\nResults saved to: {output_file}")
