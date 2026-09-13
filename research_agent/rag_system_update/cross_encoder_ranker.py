"""Cross-encoder reranking for the small RRF candidate set."""

import math

from sentence_transformers import CrossEncoder

from ..config import settings
from .data_models import RetrievedChunk


class CrossEncoderRanker:
    """Use a cross-encoder only after cheap retrieval has narrowed the search space."""

    def __init__(self):
        self._model: CrossEncoder | None = None

    def _load_model(self) -> CrossEncoder:
        if self._model is None:
            model_name = getattr(
                settings,
                "rag_reranker_model_id",
                "BAAI/bge-reranker-base",
            )
            local_only = getattr(settings, "rag_reranker_local_files_only", False)
            print(f"[RAG][RERANKER] Loading model: {model_name}")
            self._model = CrossEncoder(
                model_name,
                max_length=512,
                local_files_only=local_only,
            )
        return self._model

    def rank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not candidates:
            return []

        if not getattr(settings, "rag_reranker_enabled", True):
            print("[RAG][RERANKER] Disabled by configuration")
            return candidates

        model = self._load_model()
        pairs = [(query, candidate.document.page_content) for candidate in candidates]
        batch_size = getattr(settings, "rag_reranker_batch_size", 8)
        raw_scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

        for candidate, raw_score in zip(candidates, raw_scores):
            value = float(raw_score)
            candidate.cross_encoder_score = value
            candidate.cross_encoder_normalized = 1.0 / (1.0 + math.exp(-value))

        print(f"[RAG][CROSS ENCODER] Reranked {len(candidates)} candidates")
        return candidates
