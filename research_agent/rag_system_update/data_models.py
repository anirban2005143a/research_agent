"""Small data objects shared by the RAG pipeline."""

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document


@dataclass
class RetrievedChunk:
    """A document chunk plus retrieval information kept outside metadata."""

    chunk_id: str
    document: Document
    retrieval_score: float = 0.0
    rrf_score: float = 0.0
    rrf_normalized: float = 0.0
    cross_encoder_score: float = 0.0
    cross_encoder_normalized: float = 0.0
    final_score: float = 0.0
    dense_rank: int | None = None
    lexical_rank: int | None = None


@dataclass
class SearchResults:
    """Results returned by the retrieval stages before presentation."""

    chunks: list[RetrievedChunk] = field(default_factory=list)
    total_candidates: int = 0

    def as_dicts(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for chunk in self.chunks:
            metadata = dict(chunk.document.metadata)
            metadata["chunk_id"] = chunk.chunk_id
            metadata["score"] = round(chunk.final_score, 6)
            metadata["rrf_score"] = round(chunk.rrf_score, 6)
            metadata["cross_encoder_score"] = round(chunk.cross_encoder_score, 6)

            results.append(
                {
                    "content": chunk.document.page_content,
                    "metadata": metadata,
                }
            )
        return results
