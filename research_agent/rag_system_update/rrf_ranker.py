"""Reciprocal Rank Fusion for dense and lexical result lists."""

from .data_models import RetrievedChunk


class RRFRanker:
    """Fuse independent rankings without comparing incompatible raw scores."""

    def __init__(self, smoothing: int = 60):
        self.smoothing = smoothing

    def rank(
        self,
        dense_results: list[RetrievedChunk],
        lexical_results: list[RetrievedChunk],
        limit: int,
    ) -> list[RetrievedChunk]:
        by_id: dict[str, RetrievedChunk] = {}

        for rank, item in enumerate(dense_results, start=1):
            current = by_id.setdefault(item.chunk_id, item)
            current.dense_rank = rank
            current.rrf_score += 1.0 / (self.smoothing + rank)

        for rank, item in enumerate(lexical_results, start=1):
            current = by_id.setdefault(item.chunk_id, item)
            current.lexical_rank = rank
            if current.document is not item.document:
                current.document = item.document
            current.rrf_score += 1.0 / (self.smoothing + rank)

        ranked = sorted(by_id.values(), key=lambda item: item.rrf_score, reverse=True)
        if ranked:
            maximum = 2.0 / (self.smoothing + 1)
            for item in ranked:
                item.rrf_normalized = min(item.rrf_score / maximum, 1.0)

        ranked = ranked[:limit]
        print(
            f"[RAG][RRF] Dense={len(dense_results)}, BM25={len(lexical_results)} "
            f"-> {len(ranked)} fused candidates"
        )
        return ranked
