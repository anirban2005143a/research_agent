"""Clean orchestration layer for the research-document RAG pipeline."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from ..config import settings
from .cross_encoder_ranker import CrossEncoderRanker
from .data_models import SearchResults
from .dense_retriever import DenseRetriever
from .document_handler import DocumentHandler
from .lexical_retriever import LexicalRetriever
from .overall_ranker import OverallRanker
from .rrf_ranker import RRFRanker


class HybridRAG:
    """Coordinate ingestion and retrieval; ranking details live in dedicated classes."""

    def __init__(self, session_id: str, document_handler: DocumentHandler | None = None):
        self.session_id = session_id
        storage_dir = Path(getattr(settings, "documents_dir", "documents")) / session_id
        self.document_handler = document_handler or DocumentHandler(storage_dir)

        self.dense_retriever = DenseRetriever(session_id)
        self.lexical_retriever = LexicalRetriever()
        self.rrf_ranker = RRFRanker(smoothing=60)
        self.cross_encoder_ranker = CrossEncoderRanker()
        self.overall_ranker = OverallRanker(rrf_weight=0.4, cross_encoder_weight=0.6)

        self._rebuild_lexical_index()

    def _rebuild_lexical_index(self) -> None:
        documents = self.dense_retriever.get_all_documents()
        candidate_count = self._candidate_count()
        self.lexical_retriever.rebuild(documents, candidate_count)

    def _candidate_count(self) -> int:
        multiplier = getattr(settings, "rag_candidate_multiplier", 6)
        top_k = getattr(settings, "rag_top_k", 8)
        return max(top_k * multiplier, top_k)

    def store_document(
        self,
        file_path: str | Path,
        source_name: str | None = None,
        file_metadata: dict[str, Any] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> int:
        """Index a file once, based only on its filename in the current collection."""
        path = Path(file_path)
        source = source_name or path.name

        if self.dense_retriever.has_file(Path(source).name):
            print(f"[RAG][INDEX] Already indexed, skipping: {source}")
            return 0

        chunks = self.document_handler.prepare_file(path, filename=source)
        if not chunks:
            print(f"[RAG][INDEX] No readable text found: {source}")
            return 0

        batch_size = max(1, getattr(settings, "rag_embedding_batch_size", 16))
        self.dense_retriever.add_documents(chunks, batch_size=batch_size)

        if progress_callback:
            total = len(chunks)
            progress_callback(total, total, source)

        # BM25 is rebuilt only after ingestion, never for every query.
        self._rebuild_lexical_index()
        print(f"[RAG][INDEX] Completed: {source} | {len(chunks)} chunks")
        return len(chunks)

    def retrieve(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        """Run dense + BM25 retrieval, RRF fusion, cross-encoder reranking, and final ranking."""
        query = query.strip()
        if not query:
            return []

        final_k = k or getattr(settings, "rag_top_k", 8)
        candidate_count = max(final_k * getattr(settings, "rag_candidate_multiplier", 6), final_k)

        print(f"[RAG][QUERY] Searching for: {query}")

        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(self.dense_retriever.search, query, candidate_count)
            lexical_future = executor.submit(self.lexical_retriever.search, query, candidate_count)
            dense_results = dense_future.result()
            lexical_results = lexical_future.result()

        print(f"[RAG][RETRIEVAL] Dense={len(dense_results)}, BM25={len(lexical_results)}")

        fused = self.rrf_ranker.rank(
            dense_results,
            lexical_results,
            limit=max(final_k * 4, final_k),
        )
        reranked = self.cross_encoder_ranker.rank(query, fused)
        final_results: SearchResults = self.overall_ranker.rank(reranked, final_k)

        return final_results.as_dicts()
