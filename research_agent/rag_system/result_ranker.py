"""Candidate scoring, hybrid fusion, and reranking."""

import math
import re
from typing import Any

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from ..config import settings
from ..observability import log


_TOKEN_PATTERN = re.compile(r"[\w-]+")


def _token_list(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if math.isclose(low, high):
        return [1.0 if high != 0 else 0.0 for _ in scores]
    return [(score - low) / (high - low) for score in scores]


class ResultRanker:
    """Assign one final score after all retrieval methods produce candidates."""

    def __init__(self):
        self._reranker: CrossEncoder | None = None
        self._reranker_failed = False

    @staticmethod
    def bm25_score(document: Document, query_terms: set[str], corpus: list[Document]) -> float:
        """Score one candidate with Okapi BM25 against the candidate corpus."""
        if not query_terms:
            return 0.0
        tokenized = [_token_list(item.page_content) for item in corpus]
        document_tokens = _token_list(document.page_content)
        average_length = sum(len(tokens) for tokens in tokenized) / max(len(tokenized), 1)
        document_frequency = {
            term: sum(term in tokens for tokens in tokenized) for term in query_terms
        }
        score = 0.0
        k1 = 1.5
        b = 0.75
        for term in query_terms:
            frequency = document_tokens.count(term)
            if not frequency:
                continue
            idf = math.log(1 + (len(tokenized) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            length_ratio = len(document_tokens) / max(average_length, 1)
            score += idf * (frequency * (k1 + 1)) / (frequency + k1 * (1 - b + b * length_ratio))
        return score

    def _get_reranker(self) -> CrossEncoder | None:
        if not settings.rag_reranker_enabled or self._reranker_failed:
            return None
        if self._reranker is None:
            try:
                self._reranker = CrossEncoder(
                    settings.rag_reranker_model_id,
                    max_length=512,
                    local_files_only=settings.rag_reranker_local_files_only,
                )
            except Exception as exc:
                log(f"RAG_RERANKER | disabled_for_process | error={exc!r}")
                self._reranker_failed = True
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
            item["cross_encoder_score"] = float(score)

    def rank(
        self,
        query: str,
        dense_results: list[tuple[Document, float]],
        lexical_results: list[Document],
        query_terms: set[str],
        k: int,
    ) -> list[dict[str, Any]]:
        """Combine dense, BM25, and cross-encoder scores into final ranking."""
        candidates: dict[str, dict[str, Any]] = {}
        lexical_corpus = lexical_results
        for document, dense_score in dense_results:
            chunk_id = document.metadata.get("chunk_id", document.page_content)
            candidates.setdefault(chunk_id, {"document": document, "dense_score": 0.0, "bm25_score": 0.0})
            candidates[chunk_id]["dense_score"] = float(dense_score)
        for document in lexical_results:
            chunk_id = document.metadata.get("chunk_id", document.page_content)
            candidates.setdefault(chunk_id, {"document": document, "dense_score": 0.0, "bm25_score": 0.0})
            candidates[chunk_id]["bm25_score"] = self.bm25_score(document, query_terms, lexical_corpus)

        items = list(candidates.values())
        self._cross_encoder_scores(query, items)
        dense_scores = _normalize_scores([item["dense_score"] for item in items])
        bm25_scores = _normalize_scores([item["bm25_score"] for item in items])
        cross_scores = _normalize_scores([item.get("cross_encoder_score", 0.0) for item in items])
        cross_weight = settings.rag_cross_encoder_weight if any("cross_encoder_score" in item for item in items) else 0.0
        weight_total = settings.rag_dense_weight + settings.rag_bm25_weight + cross_weight
        for item, dense, bm25, cross in zip(items, dense_scores, bm25_scores, cross_scores):
            item["dense_normalized"] = dense
            item["bm25_normalized"] = bm25
            item["cross_encoder_normalized"] = cross
            item["final_score"] = (
                settings.rag_dense_weight * dense
                + settings.rag_bm25_weight * bm25
                + cross_weight * cross
            ) / max(weight_total, 1e-9)
        items.sort(key=lambda item: item["final_score"], reverse=True)

        selected: list[dict[str, Any]] = []
        source_counts: dict[str, int] = {}
        for item in items:
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
                    "score": round(item["final_score"], 6),
                    "final_score": round(item["final_score"], 6),
                    "dense_score": round(item["dense_score"], 6),
                    "bm25_score": round(item["bm25_score"], 6),
                    "cross_encoder_score": round(item.get("cross_encoder_score", 0.0), 6),
                    "metadata": dict(document.metadata),
                    "citation": f"{source}, page {document.metadata.get('page', '?')}"
                    + (f", section {document.metadata['headings']}" if document.metadata.get("headings") else ""),
                }
            )
            if len(selected) >= k:
                break
        return selected
