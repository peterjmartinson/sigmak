"""CLI handler for the `inspect` subcommand.

Prints a JSON summary of the ChromaDB persistence directory to stdout.
"""
from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Dict, List


def _import_chromadb() -> Any:
    """Import chromadb and return the module."""
    try:
        import chromadb  # type: ignore[import-untyped]

        return chromadb
    except Exception as exc:
        raise ImportError("chromadb is not installed or failed to import") from exc


def _create_client(persist_dir: str) -> Any:
    """Return a ChromaDB client pointing at *persist_dir*."""
    chromadb = _import_chromadb()
    # Modern API (chromadb >= 0.4)
    try:
        return chromadb.PersistentClient(path=persist_dir)
    except Exception:
        pass
    # Older Settings-based API
    try:
        from chromadb.config import Settings  # type: ignore[import-untyped]

        return chromadb.Client(
            Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir)
        )
    except Exception:
        pass
    try:
        return chromadb.Client(persist_directory=persist_dir)
    except Exception as exc:
        raise RuntimeError("Failed to create chromadb client") from exc


def _list_collection_names(client: Any) -> List[str]:
    """Return the names of all collections in *client*."""
    try:
        cols = client.list_collections()
    except Exception:
        try:
            cols = client.get_collections()
        except Exception:
            return []
    names: List[str] = []
    for c in cols:
        if isinstance(c, dict):
            name: str | None = c.get("name") or c.get("id")
        else:
            name = getattr(c, "name", None) or getattr(c, "id", None)
        if name:
            names.append(name)
    return names


def _sample_collection(client: Any, name: str, limit: int = 5) -> Dict[str, Any]:
    """Return a small sample of document ids from *name*."""
    out: Dict[str, Any] = {"name": name, "count": 0, "ids": []}
    try:
        col = client.get_collection(name=name)
        # Use .count() for the true total; ids are always returned by .get()
        out["count"] = col.count()
        sample = col.get(offset=0, limit=limit)
        out["ids"] = sample.get("ids", [])
    except Exception:
        out["error"] = traceback.format_exc()
    return out


def run(
    ticker: str | None = None,
    chroma_dir: str = "./database",
    max_sample: int = 5,
    **_: object,
) -> None:
    """Inspect the ChromaDB at *chroma_dir* and print a JSON summary.

    Parameters
    ----------
    ticker:
        Accepted for forward-compat; not used by this subcommand.
    chroma_dir:
        Path to the ChromaDB persistence directory.
    max_sample:
        Maximum number of sample rows displayed per collection.
    """
    try:
        client = _create_client(chroma_dir)
    except Exception as exc:
        print(
            f"Error: Failed to connect to ChromaDB at '{chroma_dir}': {exc}",
            file=sys.stderr,
        )
        sys.exit(2)

    available = _list_collection_names(client)
    result: Dict[str, Any] = {
        "persist_dir": chroma_dir,
        "available_collections": available,
        "collections": {},
    }
    for name in available:
        result["collections"][name] = _sample_collection(client, name, limit=max_sample)

    print(json.dumps(result, indent=2, ensure_ascii=False))
