"""Evaluate a single retrieval result set against an expected source paper."""
import os
import re
from typing import Any

from langchain_core.documents import Document

from ..data_types import RetrievedChunk, SearchResults


def source_match(metadata: dict[str, Any], expected_source: str) -> bool:
    """Return True when source or filename metadata identifies the expected file."""
    def normalize(value: str) -> str:
        filename = os.path.basename(value.replace("\\", "/"))
        return re.sub(r"[^a-z0-9]+", "_", filename.lower()).strip("_")

    expected = normalize(expected_source)
    if not expected:
        return False

    for key in ("source", "filename"):
        value = metadata.get(key)
        if value is not None and normalize(str(value)) == expected:
            return True

    return False


def evaluate_single_query(results: SearchResults, expected_source: str, k: int = 5) -> dict[str, Any]:
    """Score the matched and unmatched results returned for one query."""
    top_k_results = results.results[:k]
    matched_results = []
    unmatched_results = []

    for result in top_k_results:
        metadata = result.document.metadata
        if source_match(metadata, expected_source):
            matched_results.append(result)
        else:
            unmatched_results.append(result)

    hits = len(matched_results)
    misses = len(unmatched_results)
    precision = hits / len(top_k_results) if top_k_results else 0.0

    return {
        "precision@k": precision,
        "hits@k": hits,
        "misses@k": misses,
        "matched_results": matched_results,
        "unmatched_results": unmatched_results,
    }


if __name__ == "__main__":
    sample_chunk = RetrievedChunk(
        chunk_id="sample",
        document=Document(page_content="hello", metadata={"source": "paper_01.pdf"}),
    )
    print(evaluate_single_query(SearchResults(results=[sample_chunk]), "paper_01.pdf", k=5))
