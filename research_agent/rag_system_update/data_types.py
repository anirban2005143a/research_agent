"""Validated Pydantic types shared by the RAG pipeline."""

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field


class RetrievedChunk(BaseModel):
    """A chunk plus ranking state used between retrieval stages."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

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


class SearchResults(BaseModel):
    """Container for the ranked results returned by the RAG pipeline."""

    results: list[RetrievedChunk] = Field(default_factory=list)
