"""Lexical retrieval using LangChain's BM25Retriever."""

import uuid

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from .data_types import RetrievedChunk
from ..utils import log


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
            log("rag.bm25.index_empty | document_count=0")
            return

        self.retriever = BM25Retriever.from_documents(documents, k=candidate_count)
        log(f"rag.bm25.index_rebuilt | chunk_count={len(documents)} | candidate_count={candidate_count}")

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
