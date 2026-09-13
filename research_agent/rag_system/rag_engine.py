"""Chroma-backed document storage and hybrid retrieval."""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from ..config import settings
from ..observability import log
from .document_handler import DocumentHandler
from .result_ranker import ResultRanker


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class HybridRAG:
    """Store prepared documents in Chroma and retrieve relevant chunks."""

    def __init__(self, session_id: str, document_handler: DocumentHandler | None = None):
        self.session_id = session_id
        self.session_chroma_dir = Path(settings.chroma_dir) / session_id
        self.session_chroma_dir.mkdir(parents=True, exist_ok=True)
        self.document_handler = document_handler or DocumentHandler(
            Path(settings.documents_dir) / session_id,
            session_id=session_id,
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_id,
            cache_folder=settings.embedding_cache_dir,
            model_kwargs={"device": "cpu", "local_files_only": settings.embedding_local_files_only},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore = Chroma(
            collection_name=f"research_documents_{session_id.replace('-', '_')}",
            embedding_function=self.embeddings,
            persist_directory=str(self.session_chroma_dir),
        )
        self.ranker = ResultRanker()

    def store_document(
        self,
        file_path: str | Path,
        source_name: str | None = None,
        file_metadata: dict[str, Any] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> int:
        """Read one file, chunk it, and persist its chunks in Chroma."""
        path = Path(file_path)
        source = source_name or path.name
        source_id = stable_id(f"{self.session_id}:{source.lower()}")[:16]
        chunks = self.document_handler.prepare_file(
            path,
            source_name=source,
            file_metadata={"source_id": source_id, **(file_metadata or {})},
        )
        unique_chunks: list[Document] = []
        ids: list[str] = []
        for chunk_index, chunk in enumerate(chunks):
            chunk_hash = stable_id(f"{source_id}:{chunk.page_content}")
            chunk.metadata.update({"chunk_id": chunk_hash, "chunk_index": chunk_index, "content_hash": chunk_hash})
            ids.append(chunk_hash)
        if ids:
            existing = self.vectorstore.get(ids=ids, include=["metadatas"])
            existing_ids = set(existing.get("ids", []))
            for chunk, chunk_id in zip(chunks, ids):
                if chunk_id not in existing_ids:
                    unique_chunks.append(chunk)
        total = len(unique_chunks)
        batch_size = max(1, settings.rag_embedding_batch_size)
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_documents = unique_chunks[batch_start:batch_end]
            batch_ids = [document.metadata["chunk_id"] for document in batch_documents]
            self.vectorstore.add_documents(batch_documents, ids=batch_ids)
            for offset in range(batch_start, batch_end):
                log(f"Stored chunk {offset + 1}/{total} for {source}")
                if progress_callback:
                    progress_callback(offset + 1, total, source)
        self.document_handler.register_file(
            source,
            file_metadata={"stored_path": str(path.resolve()), **(file_metadata or {})},
            pages=len({document.metadata.get("page") for document in chunks}),
            chunk_count=len(unique_chunks),
        )
        return len(unique_chunks)

    def retrieve(self, query: str, k: int = 8) -> list[dict[str, Any]]:
        """Run bounded dense and lexical retrieval in parallel, then rank results."""
        candidate_limit = max(k * settings.rag_candidate_multiplier, k)
        terms = set(query.lower().split())

        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(
                self.vectorstore.similarity_search_with_relevance_scores,
                query,
                min(candidate_limit, 50),
            )
            lexical_future = executor.submit(
                self._lexical_search,
                terms,
                candidate_limit,
            )
            dense_docs = dense_future.result()
            lexical = lexical_future.result()
        return self.ranker.rank(query, dense_docs, lexical, terms, k)

    def _lexical_search(self, terms: set[str], limit: int) -> list[Document]:
        """Use Chroma document filters instead of loading the whole collection."""
        documents: dict[str, Document] = {}
        searchable_terms = [term for term in terms if len(term) >= 2]

        def search_term(term: str) -> dict[str, Any]:
            return self.vectorstore.get(
                where_document={"$contains": term},
                limit=limit,
                include=["documents", "metadatas"],
            )

        with ThreadPoolExecutor(max_workers=min(4, max(1, len(searchable_terms)))) as executor:
            results = executor.map(search_term, searchable_terms)
        for result in results:
            for content, metadata in zip(result.get("documents", []), result.get("metadatas", [])):
                document = Document(page_content=content, metadata=metadata or {})
                documents[document.metadata.get("chunk_id", stable_id(content))] = document
        return sorted(documents.values(), key=lambda document: self.ranker.lexical_score(document, terms), reverse=True)[:limit]
