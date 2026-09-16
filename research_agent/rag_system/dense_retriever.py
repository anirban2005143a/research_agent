"""Dense retrieval backed by Chroma and a local sentence-transformer."""

import re
import uuid
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from ..config import settings
from .data_types import RetrievedChunk


_EMBEDDING_MODEL_ID = getattr(settings, "embedding_model_id", "BAAI/bge-m3")
_EMBEDDING_CACHE_DIR = getattr(settings, "embedding_cache_dir", ".models")
_EMBEDDING_LOCAL_ONLY = getattr(settings, "embedding_local_files_only", False)

print(f"[RAG][EMBEDDING] Loading model at import: {_EMBEDDING_MODEL_ID}")
_EMBEDDINGS = HuggingFaceEmbeddings(
    model_name=_EMBEDDING_MODEL_ID,
    cache_folder=_EMBEDDING_CACHE_DIR,
    model_kwargs={"device": "cpu", "local_files_only": _EMBEDDING_LOCAL_ONLY},
    encode_kwargs={"normalize_embeddings": True}
)


def _safe_collection_name(session_id: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)
    value = value[:50] or "default"
    return f"research_documents_{value}"

def _chunk_id(document: Document) -> str:
    return str(document.id or uuid.uuid4())


class DenseRetriever:
    """Own the vector database and dense embedding operations."""

    def __init__(
        self, 
        session_id: str, 
        collection_name: str | None = None, 
        persist_directory: str | Path | None = None
    ):
        self.session_id = session_id
        self.embeddings = _EMBEDDINGS

        # 1. Resolve storage directory (fallback to config if not provided)
        if persist_directory is not None:
            self._storage_dir = Path(persist_directory)
        else:
            self._storage_dir = Path(getattr(settings, "chroma_dir", ".chroma")) / session_id
        
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        # 2. Resolve collection name (fallback to generated one if not provided)
        if collection_name is not None:
            self.collection_name = collection_name
        else:
            self.collection_name = _safe_collection_name(session_id)

        self.vector_store = self._open_collection(self.collection_name, self._storage_dir)

    def _open_collection(self, collection_name: str, storage: Path) -> Chroma:
        storage.mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(storage),
        )

    def has_file(self, file_path: str | Path) -> bool:
        """Check whether chunks for the file basename exist in the Chroma collection."""
        filename = Path(file_path).name
        result = self.vector_store.get(
            where={"source": filename},
            include=["metadatas"],
        )
        return bool(result.get("ids"))

    def delete_file(self, file_path: str | Path) -> int:
        """Delete every chunk in the current session for the basename from the given file path."""
        filename = Path(file_path).name
        result = self.vector_store.get(
            where={"source": filename},
            include=["metadatas"],
        )
        ids = result.get("ids") or []
        if ids:
            self.vector_store.delete(ids=ids)
            print(f"[RAG][VECTOR STORE] Removed {len(ids)} chunk(s) for: {filename}")
        return len(ids)

    def add_documents(self, chunks: list[Document], batch_size: int) -> None:
        total = len(chunks)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = chunks[start:end]
            print(f"[RAG][VECTOR STORE] Storing embeddings of chunks {start + 1}-{end}/{total}")
            self.vector_store.add_documents(batch)

        print(f"[RAG][VECTOR STORE] Stored {total} chunks")

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        results = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
        chunks: list[RetrievedChunk] = []
        for rank, (document, score) in enumerate(results, start=1):
            chunks.append(
                RetrievedChunk(
                    chunk_id=_chunk_id(document),
                    document=document,
                    similarity_score=float(score),
                    dense_rank=rank,
                )
            )
        return chunks

    def get_all_documents(self, session_id: str | None = None) -> list[Document]:
        """Return the indexed documents for a specific session namespace.

        When no session is supplied, the current retriever instance/session is used.
        """
        target_session = session_id or self.session_id
        store = self.vector_store if target_session == self.session_id else self._open_collection(_safe_collection_name(target_session), self._storage_dir.parent / target_session)

        result = store.get(include=["documents", "metadatas"])
        documents: list[Document] = []
        for content, metadata in zip(result.get("documents", []), result.get("metadatas", [])):
            if content is None:
                continue
            documents.append(Document(page_content=content, metadata=metadata or {}))
        return documents