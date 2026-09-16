"""Lexical retrieval using LangChain's BM25Retriever."""

import uuid

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from .data_types import RetrievedChunk


def _chunk_id(document: Document) -> str:
    return str(document.id or uuid.uuid4())


class LexicalRetriever:
    """Own the in-memory BM25 index for the current Chroma collection."""

    def __init__(self):
        self.retriever: BM25Retriever | None = None
        self.document_count = 0

    def rebuild(self, documents: list[Document], candidate_count: int) -> None:
        self.document_count = len(documents)
        if not documents:
            self.retriever = None
            print("[RAG][BM25] No documents available")
            return

        self.retriever = BM25Retriever.from_documents(documents, k=candidate_count)
        print(f"[RAG][BM25] Index ready with {len(documents)} chunks")

    def search(self, query: str, limit: int) -> list[RetrievedChunk]:
        if self.retriever is None:
            return []

        self.retriever.k = limit
        documents = self.retriever.invoke(query)
        return [
            RetrievedChunk(
                chunk_id=_chunk_id(document),
                document=document,
                lexical_rank=rank,
            )
            for rank, document in enumerate(documents, start=1)
        ]
