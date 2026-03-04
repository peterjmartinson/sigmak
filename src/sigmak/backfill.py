"""
Library module for LLM classification backfill operations.

Provides helpers for reading output/*.json result files and
persisting their LLM classifications into SQLite + ChromaDB
via DriftDetectionSystem.

These helpers were promoted from scripts/backfill_llm_cache_to_chroma.py
so that cli/backfill.py can import them without depending on the scripts/
directory.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from sigmak.drift_detection import ClassificationSource, DriftDetectionSystem
from sigmak.embeddings import EmbeddingEngine
from sigmak.llm_classifier import LLMClassificationResult
from sigmak.risk_taxonomy import validate_category

logger = logging.getLogger(__name__)


class BackfillStats:
    """Counters for a backfill run."""

    def __init__(self) -> None:
        self.files_processed: int = 0
        self.entries_found: int = 0
        self.entries_inserted: int = 0
        self.entries_skipped: int = 0
        self.errors: int = 0

    def __str__(self) -> str:
        return (
            f"Files processed: {self.files_processed}\n"
            f"Entries found:   {self.entries_found}\n"
            f"Entries inserted:{self.entries_inserted}\n"
            f"Entries skipped: {self.entries_skipped}\n"
            f"Errors:          {self.errors}"
        )


def load_output_json(file_path: Path) -> list[dict[str, Any]]:
    """Load and parse a results_*.json output file.

    Handles three formats:
    - A JSON list of risk entries
    - A dict with a ``"risks"`` key
    - A single dict entry

    Returns an empty list on any parse error.
    """
    try:
        with open(file_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "risks" in data:
                return data["risks"]
            return [data]
        logger.warning("Unexpected JSON format in %s", file_path)
        return []
    except Exception as exc:
        logger.error("Error loading %s: %s", file_path, exc)
        return []


def extract_llm_result(entry: dict[str, Any]) -> LLMClassificationResult:
    """Build an ``LLMClassificationResult`` from a risk-entry dict.

    Raises:
        ValueError: If the ``"category"`` field is missing.
    """
    if "category" not in entry:
        raise ValueError("Missing 'category' field")

    category = validate_category(entry["category"])
    confidence = float(entry.get("confidence", 0.0))
    evidence = entry.get("llm_evidence", entry.get("evidence", ""))
    rationale = entry.get("llm_rationale", entry.get("rationale", ""))
    model_version = entry.get("model_version", "unknown")
    prompt_version = entry.get("prompt_version", "1")  # default to v1 for old data

    timestamp_str = entry.get("timestamp", datetime.now().isoformat())
    try:
        timestamp = datetime.fromisoformat(timestamp_str)
    except ValueError:
        timestamp = datetime.now()

    return LLMClassificationResult(
        category=category,
        confidence=confidence,
        evidence=evidence,
        rationale=rationale,
        model_version=model_version,
        prompt_version=prompt_version,
        timestamp=timestamp,
        response_time_ms=float(entry.get("response_time_ms", 0.0)),
        input_tokens=int(entry.get("input_tokens", 0)),
        output_tokens=int(entry.get("output_tokens", 0)),
    )


def backfill_entry(
    entry: dict[str, Any],
    drift_system: DriftDetectionSystem,
    embedding_engine: EmbeddingEngine,
    dry_run: bool = True,
) -> bool:
    """Attempt to backfill a single risk entry.

    Only processes entries where ``classification_method == "llm"``.

    Returns:
        ``True`` if the entry was (or would be in dry-run) inserted,
        ``False`` if it was skipped.
    """
    risk_text: str = entry.get("risk_text", entry.get("text", ""))
    if not risk_text or not risk_text.strip():
        logger.warning("Empty risk_text, skipping entry")
        return False

    if entry.get("classification_method", "") != "llm":
        logger.debug(
            "Skipping non-LLM entry (method=%s)",
            entry.get("classification_method"),
        )
        return False

    try:
        llm_result = extract_llm_result(entry)
        embeddings = embedding_engine.encode([risk_text])
        embedding = embeddings[0].tolist()

        if dry_run:
            logger.info(
                "[DRY RUN] Would insert: category=%s, confidence=%.2f, prompt_version=%s",
                llm_result.category.value,
                llm_result.confidence,
                llm_result.prompt_version,
            )
            return True

        record_id, chroma_id = drift_system.insert_classification(
            text=risk_text,
            embedding=embedding,
            llm_result=llm_result,
            source=ClassificationSource.LLM,
            allow_duplicates=False,
        )
        logger.info("Inserted: record_id=%s, category=%s", record_id, llm_result.category.value)
        return True

    except Exception as exc:
        logger.error("Error processing entry: %s", exc)
        return False


def run_backfill(
    output_dir: str,
    db_path: str,
    chroma_path: str,
    embedding_model: str,
    dry_run: bool,
) -> BackfillStats:
    """Run the full backfill over all results_*.json files.

    Args:
        output_dir:      Directory containing results_*.json files.
        db_path:         Path to the SQLite database.
        chroma_path:     Path to the ChromaDB persistence directory.
        embedding_model: Sentence-transformer model name.
        dry_run:         When ``True`` no data is written.

    Returns:
        A ``BackfillStats`` summary.
    """
    drift_system = DriftDetectionSystem(
        db_path=db_path,
        chroma_path=chroma_path,
        embedding_model=embedding_model,
    )
    embedding_engine = EmbeddingEngine(model_name=embedding_model)

    output_path = Path(output_dir)
    json_files = sorted(output_path.glob("results_*.json"))
    logger.info("Found %d JSON files in %s", len(json_files), output_path)

    stats = BackfillStats()

    for json_file in json_files:
        logger.info("Processing: %s", json_file.name)
        stats.files_processed += 1
        entries = load_output_json(json_file)
        stats.entries_found += len(entries)

        for entry in entries:
            try:
                inserted = backfill_entry(
                    entry=entry,
                    drift_system=drift_system,
                    embedding_engine=embedding_engine,
                    dry_run=dry_run,
                )
                if inserted:
                    stats.entries_inserted += 1
                else:
                    stats.entries_skipped += 1
            except Exception as exc:
                logger.error("Failed to process entry: %s", exc)
                stats.errors += 1

    return stats
