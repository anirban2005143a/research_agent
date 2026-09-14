"""Evaluate a single retrieval result set against an expected source paper."""

from __future__ import annotations

import os
from typing import Any


def source_match(metadata: dict[str, Any], expected_source: str) -> bool:
    """Return True when the result metadata clearly points to the expected PDF source."""
    values = [str(value).lower() for value in metadata.values() if value is not None]
    blob = " ".join(values)
    expected = os.path.basename(expected_source).lower()
    return (
        expected in blob
        or os.path.splitext(expected)[0] in blob
        or any(marker in blob for marker in ["/" + expected, "\\" + expected])
    )


def evaluate_single_result(results: list[dict[str, Any]], expected_source: str, k: int = 5) -> dict[str, Any]:
    """Score one retrieved result set against one expected document source."""
    rank = None
    for index, result in enumerate(results[:k], start=1):
        metadata = result.get("metadata", {}) or {}
        if source_match(metadata, expected_source):
            rank = index
            break

    hit = rank is not None and rank <= k
    rr_score = 1.0 / rank if rank and rank <= k else 0.0

    return {
        "rank": rank,
        "hit@k": hit,
        "rr@k": rr_score,
        "latency_sec": 0.0,
        "error": None,
    }


if __name__ == "__main__":
    sample = [{"metadata": {"source": "paper_01.pdf"}, "content": "hello"}]
    print(evaluate_single_result(sample, "paper_01.pdf", k=5))
