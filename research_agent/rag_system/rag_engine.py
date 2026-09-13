"""Chroma-backed document storage and hybrid retrieval."""

import hashlib
import re
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Callable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from ..config import settings
from ..observability import log
from .document_handler import DocumentHandler


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tokens(text: str) -> list[str]:
    return re.findall(r"[\w-]+", text.lower())


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

    def _stored_documents(self) -> list[Document]:
        stored = self.vectorstore.get(include=["documents", "metadatas"])
        return [
            Document(page_content=content, metadata=metadata or {})
            for content, metadata in zip(stored.get("documents", []), stored.get("metadatas", []))
        ]

    def retrieve(self, query: str, k: int = 8) -> list[dict[str, Any]]:
        """Fuse dense, body, metadata, and typo-tolerant rankings."""
        stored_documents = self._stored_documents()
        if not stored_documents:
            return []
        vocabulary = set()
        for document in stored_documents:
            vocabulary.update(tokens(document.page_content))
            for field in ("filename", "title", "headings", "author", "year"):
                vocabulary.update(tokens(str(document.metadata.get(field, ""))))
        corrected_terms: list[str] = []
        for term in tokens(query):
            match = get_close_matches(term, vocabulary, n=1, cutoff=0.82)
            corrected_terms.append(match[0] if match else term)
        dense_docs = self.vectorstore.similarity_search_with_relevance_scores(" ".join(corrected_terms), k=min(k * 3, 30))
        terms = set(corrected_terms)

        def lexical_score(document: Document) -> float:
            body_score = len(terms & set(tokens(document.page_content))) / max(len(terms), 1)
            metadata_score = len(terms & set(tokens(" ".join(str(document.metadata.get(field, "")) for field in ("filename", "title", "headings", "author", "year"))))) / max(len(terms), 1)
            exact_metadata = sum(1 for field in ("filename", "title", "headings", "author", "year") if terms & set(tokens(str(document.metadata.get(field, "")))))
            return body_score + metadata_score * 3.0 + exact_metadata * 0.5

        lexical = sorted(stored_documents, key=lexical_score, reverse=True)[:min(k * 3, 30)]
        rankings: dict[str, dict[str, Any]] = {}
        for rank, (document, dense_score) in enumerate(dense_docs, start=1):
            chunk_id = document.metadata.get("chunk_id", stable_id(document.page_content))
            rankings.setdefault(chunk_id, {"document": document, "dense_score": float(dense_score), "lexical_score": 0.0, "rrf": 0.0})
            rankings[chunk_id]["rrf"] += 1 / (60 + rank)
        for rank, document in enumerate(lexical, start=1):
            chunk_id = document.metadata.get("chunk_id", stable_id(document.page_content))
            rankings.setdefault(chunk_id, {"document": document, "dense_score": 0.0, "lexical_score": 0.0, "rrf": 0.0})
            rankings[chunk_id]["lexical_score"] = lexical_score(document)
            rankings[chunk_id]["rrf"] += 1 / (60 + rank)
        selected: list[dict[str, Any]] = []
        per_source: dict[str, int] = {}
        for item in sorted(rankings.values(), key=lambda value: value["rrf"], reverse=True):
            document = item["document"]
            source = document.metadata.get("source", "uploaded document")
            if per_source.get(source, 0) >= 3:
                continue
            per_source[source] = per_source.get(source, 0) + 1
            selected.append({"content": document.page_content, "source": source, "page": document.metadata.get("page", "?"), "title": document.metadata.get("title", ""), "headings": document.metadata.get("headings", ""), "chunk_id": document.metadata.get("chunk_id", ""), "score": round(item["rrf"], 6), "dense_score": round(item["dense_score"], 6), "lexical_score": round(item["lexical_score"], 6), "metadata": dict(document.metadata), "citation": f"{source}, page {document.metadata.get('page', '?')}" + (f", section {document.metadata['headings']}" if document.metadata.get("headings") else "")})
            if len(selected) >= k:
                break
        return selected
