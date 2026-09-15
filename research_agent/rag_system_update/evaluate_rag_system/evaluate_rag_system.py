"""Process the source paper directory and run retrieval for evaluation cases.

This module intentionally speaks only to the active RAG engine.
It does not load documents directly from document_handler classes.
"""
import json
import sys
from pathlib import Path
from typing import Any

from research_agent.rag_system_update.rag_engine import HybridRAG
from research_agent.rag_system_update.data_types import RetrievedChunk
from .evaluate_query_result import evaluate_single_query

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SOURCE_DIR = Path(__file__).resolve().parents[1] / "Artificial_Intelligence"
SESSION_ID = "paper_eval_session"
RESULTS_PATH = Path(__file__).resolve().parent / "eval_results.json"
_CACHE: HybridRAG | None = None

def load_test_cases(cases_path: str | Path) -> list[dict[str, Any]]:
    """Load the evaluation questions from the JSON file."""
    cases_file = Path(cases_path)
    with cases_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)

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


def build_summary(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """Compute aggregate retrieval metrics for the evaluated cases."""
    total_cases = len(rows)
    total_hits = sum(int(row["hits@k"]) for row in rows)
    total_misses = sum(int(row["misses@k"]) for row in rows)
    average_precision = (
        sum(float(row["precision@k"]) for row in rows) / total_cases
        if total_cases
        else 0.0
    )

    def average(metric: str) -> float:
        return sum(float(row[metric]) for row in rows) / total_cases if total_cases else 0.0

    return {
        "num_cases": total_cases,
        "k": k,
        "average_precision@k": average_precision,
        "average_source_precision@k": average("source_precision@k"),
        "average_exact_page_precision@k": average("exact_page_precision@k"),
        "average_tolerant_page_precision@k": average("tolerant_page_precision@k"),
        "average_weighted_score@k": average("weighted_score@k"),
        "total_hits@k": total_hits,
        "total_misses@k": total_misses,
        "total_exact_page_hits@k": sum(int(row["exact_page_hits@k"]) for row in rows),
        "total_tolerant_page_hits@k": sum(int(row["tolerant_page_hits@k"]) for row in rows),
        "average_hits@k": total_hits / total_cases if total_cases else 0.0,
        "average_misses@k": total_misses / total_cases if total_cases else 0.0,
    }


def write_results(report: dict[str, Any], results_path: str | Path = RESULTS_PATH) -> None:
    """Persist the evaluation report to the standard results JSON file."""
    with Path(results_path).open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


def evaluate_test_case(
    test_case: dict[str, Any],
    rag: HybridRAG,
    k: int = 5,
) -> dict[str, Any]:
    """Run one evaluation test_case: retrieve results for the question, then score the result set."""
    query = test_case["question"]
    expected_source = test_case["expected_source"]
    expected_page = test_case.get("expected_page")
    relevant_pages = test_case.get("relevant_pages", [])
    results = rag.retrieve(query, k=k)
    score = evaluate_single_query(
        results,
        expected_source,
        expected_page=expected_page,
        relevant_pages=relevant_pages,
    )

    return {
        "id": test_case.get("id"),
        "question": query,
        "expected_source": expected_source,
        "expected_page": expected_page,
        "relevant_pages": relevant_pages,
        "precision@k": score["precision@k"],
        "source_precision@k": score["source_precision@k"],
        "exact_page_precision@k": score["exact_page_precision@k"],
        "tolerant_page_precision@k": score["tolerant_page_precision@k"],
        "weighted_score@k": score["weighted_score@k"],
        "hits@k": score["hits@k"],
        "misses@k": score["misses@k"],
        "exact_page_hits@k": score["exact_page_hits@k"],
        "tolerant_page_hits@k": score["tolerant_page_hits@k"],
        "matched_results": [result.model_dump() for result in score["matched_results"]],
        "unmatched_results": [result.model_dump() for result in score["unmatched_results"]],
        "exact_page_results": [result.model_dump() for result in score["exact_page_results"]],
        "tolerant_page_results": [result.model_dump() for result in score["tolerant_page_results"]],
    }


def evaluate_all_cases(
    test_cases_path: str | Path,
    rag: HybridRAG,
    k: int = 5,
) -> dict[str, Any]:
    """Evaluate every query using an already-indexed RAG engine and write the final report."""
    test_cases = load_test_cases(test_cases_path)

    rows: list[dict[str, Any]] = []
    for test_case in test_cases:
        rows.append(evaluate_test_case(test_case=test_case, rag=rag, k=k))

    summary = build_summary(rows, k=k)
    report = {"summary": summary, "test_cases": rows}
    write_results(report, results_path=RESULTS_PATH)
    return report


if __name__ == "__main__":
    source_dir = SOURCE_DIR
    cases_path = Path(__file__).resolve().parent / "eval_cases_temp.json"

    rag = process_source_directory(source_dir=source_dir, session_id=SESSION_ID)
    result = evaluate_all_cases(
        test_cases_path=cases_path,
        rag=rag,
        k=5,
    )

    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print(f"\nResults written to {RESULTS_PATH}")
