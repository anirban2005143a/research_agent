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
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> int:
        """Index a single file path only. Directory expansion belongs in the CLI/test harness."""
        path = Path(file_path)
        if path.is_dir():
            raise ValueError(
                "HybridRAG.store_document expects a single file path, not a directory. "
                "Use the main/test entrypoint to iterate files from a directory."
            )
        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {path}")

        source = path.name
        session_path = self.document_handler.storage_dir / source

        if path.resolve() != session_path.resolve():
            if session_path.exists():
                print(f"[RAG][INDEX] Replacing prior session copy: {source}")
                self.document_handler.remove_file(source)
            self.document_handler.save_upload(source, path.read_bytes())
            path = session_path

        if self.dense_retriever.has_file(source):
            print(f"[RAG][INDEX] Replacing existing Chroma entries for: {source}")
            self.dense_retriever.delete_file(source)

        chunks = self.document_handler.prepare_file(path)
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
