"""Document ingestion and hybrid retrieval components."""

from .document_handler import DocumentHandler, SUPPORTED_EXTENSIONS
from .result_ranker import ResultRanker

__all__ = ["DocumentHandler", "HybridRAG", "ResultRanker", "SUPPORTED_EXTENSIONS"]


def __getattr__(name: str):
	if name == "HybridRAG":
		from .rag_engine import HybridRAG

		return HybridRAG
	raise AttributeError(name)