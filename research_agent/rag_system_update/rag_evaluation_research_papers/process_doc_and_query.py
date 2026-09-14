"""Adapter that evaluates the active HybridRAG implementation against the paper set."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research_agent.rag_system_update.rag_engine import HybridRAG

_EVAL_SESSION_ID = "paper_eval_session"
_EVAL_PDF_DIR = Path(__file__).resolve().parents[1] / "Artificial_Intelligence"
_CACHE: HybridRAG | None = None


def _load_rag() -> HybridRAG:
    """Create a single shared RAG instance and index the evaluation papers once."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    rag = HybridRAG(session_id=_EVAL_SESSION_ID)
    for file_path in sorted(_EVAL_PDF_DIR.glob("*.pdf")):
        rag.store_document(file_path)
    _CACHE = rag
    return rag


def retrieve(query: str, k: int = 5, mode: str = "hybrid") -> list[dict[str, Any]]:
    """Return evaluated retrieval results from the active RAG pipeline."""
    rag = _load_rag()
    mode_name = (mode or "hybrid").lower()

    if mode_name == "vector_only":
        results = rag.dense_retriever.search(query, limit=k)
        return [
            {"content": item.document.page_content, "metadata": dict(item.document.metadata or {})}
            for item in results
        ]

    if mode_name == "bm25":
        results = rag.lexical_retriever.search(query, limit=k)
        return [
            {"content": item.document.page_content, "metadata": dict(item.document.metadata or {})}
            for item in results
        ]

    return rag.retrieve(query, k=k)


def normalize_results(results):
    """Normalize either dict payloads or LangChain Document objects into a uniform shape."""
    out = []
    for item in results:
        if isinstance(item, dict):
            out.append(item)
            continue

        metadata = getattr(item, "metadata", {}) or {}
        content = getattr(item, "page_content", str(item))
        out.append({"content": content, "metadata": metadata})
    return out
