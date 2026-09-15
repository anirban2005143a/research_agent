"""Evaluate retrieved chunks against an expected source paper and page."""
import os
import re
from typing import Any

from langchain_core.documents import Document

from ..data_types import RetrievedChunk, SearchResults


DEFAULT_PAGE_TOLERANCE = 2
EXACT_PAGE_WEIGHT = 0.6
TOLERANT_PAGE_WEIGHT = 0.4


def _normalize_filename(value: str) -> str:
    """Normalize a source value so paths and filenames compare consistently."""
    filename = os.path.basename(value.replace("\\", "/"))
    return re.sub(r"[^a-z0-9]+", "_", filename.lower()).strip("_")


def source_match(metadata: dict[str, Any], expected_source: str) -> bool:
    """Return True when source or filename metadata identifies the expected file."""
    expected = _normalize_filename(expected_source)
    if not expected:
        return False

    for key in ("source", "filename"):
        value = metadata.get(key)
        if value is not None and _normalize_filename(str(value)) == expected:
            return True

    return False


def _page_number(metadata: dict[str, Any]) -> int | None:
    """Read a numeric page from either supported metadata key."""
    for key in ("page_number", "page"):
        value = metadata.get(key)
        try:
            page = int(value)
        except (TypeError, ValueError):
            continue
        if page > 0:
            return page
    return None


def page_match(
    metadata: dict[str, Any],
    expected_page: int | None = None,
    relevant_pages: list[int] | None = None,
    tolerance: int = DEFAULT_PAGE_TOLERANCE,
) -> str:
    """Classify a page as exact, tolerant, mismatch, or unavailable."""
    target_pages = {page for page in (relevant_pages or []) if page > 0}
    if expected_page is not None and expected_page > 0:
        target_pages.add(expected_page)
    if not target_pages:
        return "not_evaluated"

    actual_page = _page_number(metadata)
    if actual_page is None:
        return "unavailable"
    if actual_page in target_pages:
        return "exact"
    if any(abs(actual_page - target_page) <= tolerance for target_page in target_pages):
        return "tolerant"
    return "mismatch"


def evaluate_single_query(
    results: SearchResults,
    expected_source: str,
    expected_page: int | None = None,
    relevant_pages: list[int] | None = None,
    page_tolerance: int = DEFAULT_PAGE_TOLERANCE,
) -> dict[str, Any]:
    """Score source relevance first, then exact and near page relevance.

    A source match contributes 0.70. A page in ``relevant_pages`` or equal to
    ``expected_page`` adds 0.30, and a page within ``page_tolerance`` adds 0.15.
    This keeps source relevance primary while distinguishing exact page hits
    from useful nearby context.
    """
    if page_tolerance < 0:
        raise ValueError("page_tolerance must be non-negative")

    top_k_results = results.results
    matched_results = []
    unmatched_results = []
    exact_page_results = []
    tolerant_page_results = []
    source_match_count = 0
    exact_page_count = 0
    tolerant_page_count = 0
    weighted_scores = []

    for result in top_k_results:
        metadata = result.document.metadata
        is_source_match = source_match(metadata, expected_source)
        result_page_match = page_match(
            metadata,
            expected_page=expected_page,
            relevant_pages=relevant_pages,
            tolerance=page_tolerance,
        )
        if is_source_match:
            source_match_count += 1
            if result_page_match == "exact":
                exact_page_results.append(result)
                exact_page_count += 1

                matched_results.append(result)

            elif result_page_match == "tolerant":
                tolerant_page_results.append(result)
                tolerant_page_count += 1

                matched_results.append(result)

            else :
                unmatched_results.append(result)
        else:
            unmatched_results.append(result)

        weighted_scores.append(
            (EXACT_PAGE_WEIGHT if is_source_match and result_page_match == "exact" else 0.0)
            + (TOLERANT_PAGE_WEIGHT if is_source_match and result_page_match == "tolerant" else 0.0)
        )

    result_count = len(top_k_results)
    total_relevent_count = exact_page_count + tolerant_page_count
    precision = total_relevent_count / result_count if result_count else 0.0
    source_precision = source_match_count / result_count if result_count else 0.0
    exact_page_precision = exact_page_count / result_count if result_count else 0.0
    tolerant_page_precision = tolerant_page_count / result_count if result_count else 0.0
    weighted_score = sum(weighted_scores) / result_count if result_count else 0.0

    return {
        "precision@k": precision,
        "source_precision@k": source_precision,
        "exact_page_precision@k": exact_page_precision,
        "tolerant_page_precision@k": tolerant_page_precision,
        "weighted_score@k": weighted_score,
        "hits@k": total_relevent_count,
        "misses@k": result_count - total_relevent_count,
        "exact_page_hits@k": exact_page_count,
        "tolerant_page_hits@k": tolerant_page_count,
        "matched_results": matched_results,
        "unmatched_results": unmatched_results,
        "exact_page_results": exact_page_results,
        "tolerant_page_results": tolerant_page_results,
    }


if __name__ == "__main__":
    sample_chunk = RetrievedChunk(
        chunk_id="sample",
        document=Document(page_content="hello", metadata={"source": "paper_01.pdf", "page_number": 3}),
    )
    print(evaluate_single_query(SearchResults(results=[sample_chunk]), "paper_01.pdf", expected_page=3))
