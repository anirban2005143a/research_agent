# Research Agent RAG System

This version intentionally keeps the RAG system simple at the boundaries and strict about separation of responsibilities.

## Architecture

```text
Uploaded file
    |
    v
DocumentHandler
    |  load + clean + title/filename/heading + chunk
    v
DenseRetriever --------------------> Chroma + BGE embeddings
    |
    +------------------------------> LexicalRetriever -> LangChain BM25
                                      |
                                      v
                               RRFRanker
                                      |
                                      v
                              CrossEncoderRanker
                                      |
                                      v
                               OverallRanker
                                      |
                                      v
                                  final top-k
```

## Responsibilities

- `document_handler.py`: file loading, simple metadata, chunking, upload storage.
- `dense_retriever.py`: Chroma and embedding model only.
- `lexical_retriever.py`: BM25Retriever only.
- `rrf_ranker.py`: Reciprocal Rank Fusion only.
- `cross_encoder_ranker.py`: cross-encoder inference only.
- `overall_ranker.py`: final score and final selection only.
- `rag_engine.py`: orchestration only.
- `data_models.py`: shared typed result objects.

## Metadata

Chunk metadata is deliberately limited to:

- `filename`
- `title`
- `heading`

Session information is represented by the Chroma collection/storage namespace, not copied into every chunk.

There is no document registry, file SHA-256, SQLite database, or hash-based metadata.

## Hybrid ranking

Dense retrieval and BM25 produce ranked lists. RRF combines those lists using:

`RRF(d) = sum(1 / (60 + rank))`

The fused candidates are then reranked by `BAAI/bge-reranker-base`.

The final score is a **weighted harmonic mean** of normalized RRF evidence and sigmoid-transformed cross-encoder relevance:

`final = 1 / (0.4 / rrf + 0.6 / cross_encoder)`

The harmonic mean is intentionally conservative: a candidate needs support from both stages instead of winning through a simple addition of incompatible score scales.

## Recommended CPU models

- Embedding: `BAAI/bge-base-en-v1.5`
- Cross encoder: `BAAI/bge-reranker-base`

For your 12-thread Ryzen / 16 GB RAM machine, keep reranking limited to a small fused candidate set. Start with:

- dense candidates: `k * 6`
- BM25 candidates: `k * 6`
- RRF candidates: `k * 4`
- final results: `8`

## Configuration

The implementation is compatible with the existing project `settings` object. Optional RAG-specific environment variables are read through these existing settings names where available:

- `RAG_CHUNK_SIZE`
- `RAG_CHUNK_OVERLAP`
- `RAG_EMBEDDING_BATCH_SIZE`
- `RAG_TOP_K`
- `RAG_CANDIDATE_MULTIPLIER`
- `RAG_RERANKER_MODEL_ID`
- `RAG_RERANKER_LOCAL_FILES_ONLY`

If no dedicated embedding model setting exists, the RAG system defaults to `BAAI/bge-base-en-v1.5`.

Install/ensure:

```bash
pip install langchain-chroma langchain-huggingface langchain-community langchain-text-splitters sentence-transformers rank-bm25 chromadb pypdf python-docx openpyxl python-pptx beautifulsoup4
```

The old Chroma collection should be rebuilt once because this version changes the embedding model and retrieval/indexing design.
