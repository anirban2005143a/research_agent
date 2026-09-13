"""Dense retrieval backed by Chroma and a local sentence-transformer."""

import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from ..config import settings
from .data_models import RetrievedChunk


def _safe_collection_name(session_id: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)
    value = value[:50] or "default"
    return f"research_documents_{value}"


def _chunk_id(document: Document) -> str:
    metadata = document.metadata
    return f"{metadata.get('filename', '')}::{metadata.get('heading', '')}::{document.page_content}"


class DenseRetriever:
    """Own the vector database and dense embedding operations."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        embedding_model = getattr(
            settings,
            "rag_embedding_model_id",
            "BAAI/bge-base-en-v1.5",
        )
        cache_dir = getattr(settings, "embedding_cache_dir", ".models")
        local_only = getattr(settings, "embedding_local_files_only", False)

        print(f"[RAG][EMBEDDING] Model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            cache_folder=cache_dir,
            model_kwargs={"device": "cpu", "local_files_only": local_only},
            encode_kwargs={"normalize_embeddings": True},
        )

        self._storage_dir = Path(getattr(settings, "chroma_dir", ".chroma")) / session_id
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store = self._open_collection(session_id)

    def _open_collection(self, session_id: str) -> Chroma:
        storage = Path(getattr(settings, "chroma_dir", ".chroma")) / session_id
        storage.mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name=_safe_collection_name(session_id),
            embedding_function=self.embeddings,
            persist_directory=str(storage),
        )

    def has_file(self, filename: str) -> bool:
        """Check the Chroma collection directly; no registry or file hash is needed."""
        result = self.vector_store.get(
            where={"filename": filename},
            include=["metadatas"],
        )
        return bool(result.get("ids"))

    def add_documents(self, chunks: list[Document], batch_size: int) -> None:
        total = len(chunks)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = chunks[start:end]
            ids = [f"{document.metadata['filename']}::chunk::{start + index}" for index, document in enumerate(batch)]
            print(f"[RAG][EMBEDDING] Embedding chunks {start + 1}-{end}/{total}")
            self.vector_store.add_documents(batch, ids=ids)

        print(f"[RAG][VECTOR STORE] Stored {total} chunks")

    def search(self, query: str, limit: int) -> list[RetrievedChunk]:
        results = self.vector_store.similarity_search_with_relevance_scores(query, k=limit)
        chunks: list[RetrievedChunk] = []
        for rank, (document, score) in enumerate(results, start=1):
            chunks.append(
                RetrievedChunk(
                    chunk_id=_chunk_id(document),
                    document=document,
                    retrieval_score=float(score),
                    dense_rank=rank,
                )
            )
        return chunks

    def get_all_documents(self, session_id: str | None = None) -> list[Document]:
        """Return the indexed documents for a specific session namespace.

        When no session is supplied, the current retriever instance/session is used.
        """
        target_session = session_id or self.session_id
        store = self.vector_store if target_session == self.session_id else self._open_collection(target_session)

        result = store.get(include=["documents", "metadatas"])
        documents: list[Document] = []
        for content, metadata in zip(result.get("documents", []), result.get("metadatas", [])):
            if content is None:
                continue
            documents.append(Document(page_content=content, metadata=metadata or {}))
        return documents
