"""Cross-encoder reranking for the small RRF candidate set."""

import math

from sentence_transformers import CrossEncoder

from ..config import settings
from .data_types import RetrievedChunk

_CROSS_ENCODER_MODEL_ID = getattr(
    settings, "rag_corss_encoder_model_id", "BAAI/bge-reranker-base"
)
_CROSS_ENCODER_LOCAL_ONLY = getattr(settings, "rag_corss_encoder_local_files_only", False)
_CROSS_ENCODER_CACHE_DIR = getattr(settings, "rag_cross_encoder_cache_dir", False)

print(f"[RAG][CROSS_ENCODER] Loading model at import: {_CROSS_ENCODER_MODEL_ID}")
_CROSS_ENCODER_MODEL = CrossEncoder(
    model_name_or_path=_CROSS_ENCODER_MODEL_ID,
    cache_folder=_CROSS_ENCODER_CACHE_DIR,
    max_length=512,
    local_files_only=_CROSS_ENCODER_LOCAL_ONLY,
)


class CrossEncoderRanker:
    """Use a cross-encoder only after cheap retrieval has narrowed the search space."""

    def __init__(self):
        self._model: CrossEncoder = _CROSS_ENCODER_MODEL

    def _load_model(self) -> CrossEncoder:
        return self._model

    def rank(
        self, query: str, candidates: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []

        if not getattr(settings, "rag_corss_encoder_enabled", True):
            print("[RAG][CROSS_ENCODER] Disabled by configuration")
            return candidates

        model = self._load_model()
        pairs = [(query, candidate.document.page_content) for candidate in candidates]
        batch_size = getattr(settings, "rag_corss_encoder_batch_size", 8)
        raw_scores = model.predict(
            pairs, batch_size=batch_size, show_progress_bar=False
        )

        for candidate, raw_score in zip(candidates, raw_scores):
            value = float(raw_score)
            candidate.cross_encoder_score = value
            candidate.cross_encoder_normalized = 1.0 / (1.0 + math.exp(-value))

        print(f"[RAG][CROSS ENCODER] Reranked {len(candidates)} candidates")
        return candidates
