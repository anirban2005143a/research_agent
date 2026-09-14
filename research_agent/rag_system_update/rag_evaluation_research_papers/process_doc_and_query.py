"""Process the source paper directory and run retrieval for evaluation cases.

This module intentionally speaks only to the active RAG engine.
It does not load documents directly from document_handler classes.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research_agent.rag_system_update.rag_engine import HybridRAG
from evaluate_retrieval import evaluate_single_result

SOURCE_DIR = Path(__file__).resolve().parents[1] / "Artificial_Intelligence"
SESSION_ID = "paper_eval_session"
RESULTS_PATH = Path(__file__).resolve().parent / "eval_results.json"
_CACHE: HybridRAG | None = None


def process_source_directory(
    source_dir: str | Path = SOURCE_DIR,
    session_id: str = SESSION_ID,
) -> HybridRAG:
    """Index every PDF in the given source directory through the active HybridRAG engine."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    rag = HybridRAG(session_id=session_id)
    source_path = Path(source_dir)

    # for file_path in sorted(source_path.glob("*.pdf")):
    #     rag.store_document(file_path)

    _CACHE = rag
    return rag


def retrieve(
    query: str,
    rag: HybridRAG,
    k: int = 5,
    mode: str = "hybrid",
) -> list[dict[str, Any]]:
    """Run a query through an already-indexed RAG engine and normalize the raw results."""
    mode_name = (mode or "hybrid").lower()

    if mode_name == "vector_only":
        raw_results = rag.dense_retriever.search(query, limit=k)
    elif mode_name == "bm25":
        raw_results = rag.lexical_retriever.search(query, limit=k)
    else:
        raw_results = rag.retrieve(query, k=k)

    normalized: list[dict[str, Any]] = []
    for item in raw_results:
        if isinstance(item, dict):
            normalized.append(item)
            continue

        metadata = getattr(item, "metadata", {}) or {}
        content = getattr(item, "page_content", str(item))
        normalized.append({"content": content, "metadata": dict(metadata)})

    return normalized


def evaluate_case(
    case: dict[str, Any],
    rag: HybridRAG,
    k: int = 5,
    mode: str = "hybrid",
) -> dict[str, Any]:
    """Run one evaluation case: retrieve results for the question, then score the result set."""
    query = case["question"]
    expected_source = case["expected_source"]
    started_at = time.perf_counter()
    results = retrieve(query=query, rag=rag, k=k, mode=mode)
    latency_sec = time.perf_counter() - started_at
    score = evaluate_single_result(results, expected_source, k=k)
    score["latency_sec"] = latency_sec

    return {
        "id": case.get("id"),
        "question": query,
        "expected_source": expected_source,
        "relevant_pages": case.get("relevant_pages", []),
        "rank": score["rank"],
        "hit@k": score["hit@k"],
        "rr@k": score["rr@k"],
        "latency_sec": score.get("latency_sec", 0.0),
        "results": [{
            "metadata": result.get("metadata", {}),
            "content": result.get("content", "")[:500],
        } for result in results],
        "error": score.get("error"),
    }


def load_evaluation_cases(cases_path: str | Path) -> list[dict[str, Any]]:
    """Load the evaluation questions from the JSON file."""
    cases_file = Path(cases_path)
    with cases_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_summary(rows: list[dict[str, Any]], k: int, mode: str) -> dict[str, Any]:
    """Compute the aggregate metrics for the evaluated cases."""
    total_cases = len(rows)
    hits = sum(1 for row in rows if row["hit@k"]) if rows else 0
    mrr = sum(float(row["rr@k"]) for row in rows) / total_cases if total_cases else 0.0
    avg_latency = sum(float(row["latency_sec"]) for row in rows) / total_cases if total_cases else 0.0

    return {
        "num_cases": total_cases,
        "k": k,
        "mode": mode,
        "Recall@K": hits / total_cases if total_cases else 0.0,
        "MRR@K": mrr,
        "avg_latency_sec": avg_latency,
        "errors": sum(1 for row in rows if row.get("error")),
    }


def write_results(report: dict[str, Any], results_path: str | Path = RESULTS_PATH) -> None:
    """Persist the evaluation report to the standard results JSON file."""
    with Path(results_path).open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


def evaluate_all_cases(
    cases_path: str | Path,
    rag: HybridRAG,
    k: int = 5,
    mode: str = "hybrid",
) -> dict[str, Any]:
    """Evaluate every query using an already-indexed RAG engine and write the final report."""
    cases = load_evaluation_cases(cases_path)

    rows: list[dict[str, Any]] = []
    for case in cases:
        rows.append(evaluate_case(case=case, rag=rag, k=k, mode=mode))

    summary = build_summary(rows, k=k, mode=mode)
    report = {"summary": summary, "cases": rows}
    write_results(report, results_path=RESULTS_PATH)
    return report


if __name__ == "__main__":
    source_dir = SOURCE_DIR
    cases_path = Path(__file__).resolve().parent / "eval_cases.json"

    rag = process_source_directory(source_dir=source_dir, session_id=SESSION_ID)
    result = evaluate_all_cases(
        cases_path=cases_path,
        rag=rag,
        k=5,
        mode="hybrid",
    )

    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print(f"\nResults written to {RESULTS_PATH}")
