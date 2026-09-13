"""Final score calculation and result selection."""

from .data_models import RetrievedChunk, SearchResults


class OverallRanker:
    """Combine RRF and cross-encoder evidence using a weighted harmonic mean."""

    def __init__(self, rrf_weight: float = 0.4, cross_encoder_weight: float = 0.6):
        total = rrf_weight + cross_encoder_weight
        self.rrf_weight = rrf_weight / total
        self.cross_encoder_weight = cross_encoder_weight / total

    def _final_score(self, candidate: RetrievedChunk) -> float:
        """Weighted harmonic mean rewards agreement between both ranking signals.

        RRF measures agreement between independent retrieval systems.
        The cross-encoder measures query-passage relevance directly.
        A harmonic mean is deliberately conservative: a candidate cannot get a
        high final score from only one signal while the other signal is poor.
        """
        rrf = max(candidate.rrf_normalized, 1e-8)
        cross = max(candidate.cross_encoder_normalized, 1e-8)
        denominator = (self.rrf_weight / rrf) + (self.cross_encoder_weight / cross)
        return 1.0 / denominator

    def rank(self, candidates: list[RetrievedChunk], limit: int) -> SearchResults:
        for candidate in candidates:
            candidate.final_score = self._final_score(candidate)

        candidates.sort(key=lambda item: item.final_score, reverse=True)
        selected = candidates[:limit]
        print(f"[RAG][FINAL RANKING] Selected {len(selected)} results")
        return SearchResults(chunks=selected, total_candidates=len(candidates))
