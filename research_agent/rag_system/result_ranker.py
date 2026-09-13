"""Candidate fusion and reranking for hybrid retrieval."""

from typing import Any

from langchain_core.documents import Document

from ..config import settings
from ..observability import log


def _tokens(text: str) -> set[str]:
    import re

    return set(re.findall(r"[\w-]+", text.lower()))


class ResultRanker:
    """Fuse retrieval strategies and optionally rerank their bounded candidate set."""

    def __init__(self):
        self._reranker = None

    @staticmethod
    def lexical_score(document: Document, terms: set[str]) -> float:
        body_terms = _tokens(document.page_content)
        metadata_terms = _tokens(
            " ".join(
                str(document.metadata.get(field, ""))
                for field in ("filename", "title", "headings", "author", "year")
            )
        )
        exact_metadata = sum(
            1
            for field in ("filename", "title", "headings", "author", "year")
            if terms & _tokens(str(document.metadata.get(field, "")))
        )
        return (
            len(terms & body_terms) / max(len(terms), 1)
            + 3.0 * len(terms & metadata_terms) / max(len(terms), 1)
            + 0.5 * exact_metadata
        )

    def _get_reranker(self):
        if not settings.rag_reranker_enabled or self._reranker is False:
            return None
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(
                    settings.rag_reranker_model_id,
                    max_length=512,
                    local_files_only=settings.rag_reranker_local_files_only,
                )
            except Exception as exc:
                log(f"RAG_RERANKER | disabled_for_process | error={exc!r}")
                self._reranker = False
        return self._reranker

    def _cross_encoder_scores(self, query: str, candidates: list[dict[str, Any]]) -> None:
        reranker = self._get_reranker()
        if reranker is None or not candidates:
            return
        scores = reranker.predict(
            [(query, item["document"].page_content) for item in candidates],
            batch_size=settings.rag_embedding_batch_size,
            show_progress_bar=False,
        )
        for item, score in zip(candidates, scores):
            item["reranker_score"] = float(score)

    def rank(
        self,
        query: str,
        dense_results: list[tuple[Document, float]],
        lexical_results: list[Document],
        terms: set[str],
        k: int,
    ) -> list[dict[str, Any]]:
        """Fuse dense and lexical rankings, rerank, and enforce source diversity."""
        rankings: dict[str, dict[str, Any]] = {}
        for rank, (document, dense_score) in enumerate(dense_results, start=1):
            chunk_id = document.metadata.get("chunk_id", document.page_content)
            rankings.setdefault(
                chunk_id,
                {"document": document, "dense_score": float(dense_score), "lexical_score": 0.0, "rrf": 0.0},
            )
            rankings[chunk_id]["rrf"] += 1 / (60 + rank)
        for rank, document in enumerate(lexical_results, start=1):
            chunk_id = document.metadata.get("chunk_id", document.page_content)
            rankings.setdefault(
                chunk_id,
                {"document": document, "dense_score": 0.0, "lexical_score": 0.0, "rrf": 0.0},
            )
            rankings[chunk_id]["lexical_score"] = self.lexical_score(document, terms)
            rankings[chunk_id]["rrf"] += 1 / (60 + rank)

        candidates = list(rankings.values())
        self._cross_encoder_scores(query, candidates)
        if any("reranker_score" in item for item in candidates):
            candidates.sort(key=lambda item: item["reranker_score"], reverse=True)
        else:
            candidates.sort(key=lambda item: item["rrf"], reverse=True)

        selected: list[dict[str, Any]] = []
        source_counts: dict[str, int] = {}
        for item in candidates:
            document = item["document"]
            source = document.metadata.get("source", "uploaded document")
            if source_counts.get(source, 0) >= 3:
                continue
            source_counts[source] = source_counts.get(source, 0) + 1
            selected.append(
                {
                    "content": document.page_content,
                    "source": source,
                    "page": document.metadata.get("page", "?"),
                    "title": document.metadata.get("title", ""),
                    "headings": document.metadata.get("headings", ""),
                    "chunk_id": document.metadata.get("chunk_id", ""),
                    "score": round(item["rrf"], 6),
                    "dense_score": round(item["dense_score"], 6),
                    "lexical_score": round(item["lexical_score"], 6),
                    "reranker_score": round(item.get("reranker_score", 0.0), 6),
                    "metadata": dict(document.metadata),
                    "citation": f"{source}, page {document.metadata.get('page', '?')}"
                    + (f", section {document.metadata['headings']}" if document.metadata.get("headings") else ""),
                }
            )
            if len(selected) >= k:
                break
        return selected
