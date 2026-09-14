# Research Agent

Research Agent is a research-focused assistant that combines LLM-based planning with a local document retrieval pipeline. The project supports evidence-based research questions, optional document grounding, and hybrid retrieval over uploaded files and external sources.

## Current RAG implementation

The active document pipeline is the refactored update package under [research_agent/rag_system_update](research_agent/rag_system_update). It keeps the core responsibilities clean:

- generic file validation and chunking belong to `DocumentHandler`
- PDF-specific parsing and heading extraction belong to `PDFDocumentHandler`
- document indexing and retrieval orchestration belong to `HybridRAG`
- ranking is separated into dedicated retriever and ranker components

The old `rag_system` package still exists as a separate legacy implementation, but the active flow for the current project is the update package.

## Supported document types

The current `DocumentHandler` only allows these extensions:

- `.pdf`
- `.md`
- `.ppt`
- `.pptx`
- `.docx`
- `.txt`

Anything outside that allowlist is rejected during file preparation.

## Architecture

```text
Uploaded file
    |
    v
DocumentHandler
    |-- validates extension
    |-- chooses PDF-specific handler when needed
    |-- loads file contents
    |-- cleans page/text content
    |-- extracts metadata
    |-- generates chunks
    v
HybridRAG / rag_engine
    |-- stores chunks in Chroma
    |-- embeds chunks with local model
    |-- retrieves dense + BM25 candidates
    |-- reranks and returns top-k results
```

## RAG update package layout

```text
research_agent/
├── rag_system_update/
│   ├── __init__.py
│   ├── README.md
│   ├── main.py
│   ├── rag_engine.py
│   ├── data_models.py
│   ├── dense_retriever.py
│   ├── lexical_retriever.py
│   ├── overall_ranker.py
│   ├── rrf_ranker.py
│   ├── cross_encoder_ranker.py
│   └── document_handler/
│       ├── __init__.py
│       ├── document_handler.py
│       └── pdf_handler.py
```

### Document handler responsibilities

The update package uses a parent/child structure:

- `DocumentHandler` handles shared loading, validation, storage, and chunk generation
- `PDFDocumentHandler` adds PDF-specific parsing, TOC extraction, title handling, and page heading logic

This keeps the generic path simple while preserving PDF-specific behavior in one child class.

## Retrieval flow

1. A file is saved under the session storage directory.
2. `prepare_file()` resolves the proper document handler.
3. The file is loaded as LangChain `Document` objects.
4. Each document is cleaned and metadata is attached.
5. `generate_chunks()` splits the content into retrieval chunks.
6. `HybridRAG.store_document()` indexes those chunks in Chroma.
7. Retrieval combines dense and lexical signals, reranks the results, and returns the final top-k set.

## Local indexing and retrieval

The active RAG system is meant to run locally with the project settings object controlling storage, chunk size, embedding model, batch size, and retrieval options.

Typical project settings include:

- `DOCUMENTS_DIR`
- `CHROMA_DIR`
- `RAG_CHUNK_SIZE`
- `RAG_CHUNK_OVERLAP`
- `RAG_EMBEDDING_BATCH_SIZE`
- `RAG_TOP_K`
- `RAG_CANDIDATE_MULTIPLIER`

The update package is designed so that the app can index uploaded files and later recall the most relevant chunks without depending on a hard-coded path or ad hoc loader logic.

## Running the document pipeline

From the project root:

```powershell
python -m research_agent.rag_system_update.main
```

This starts the update-package RAG flow and exercises the active document handler implementation.

## Evaluation pipeline

The project also includes a dedicated research-paper evaluation pack under [research_agent/rag_system_update/rag_evaluation_research_papers](research_agent/rag_system_update/rag_evaluation_research_papers), with its own documentation in [research_agent/rag_system_update/rag_evaluation_research_papers/README.md](research_agent/rag_system_update/rag_evaluation_research_papers/README.md).

This folder is used to evaluate the live retrieval pipeline on a fixed set of AI research PDFs stored under:

```text
research_agent/rag_system_update/Artificial_Intelligence/
```

The evaluation flow is:

1. process the source directory and index all PDFs through `HybridRAG`
2. load the evaluation cases from `eval_cases.json`
3. run each question against the live retriever
4. score whether the expected paper appears in the top-k results
5. write the summary and per-case results to `eval_results.json`

Run the evaluation from the project root with:

```powershell
python research_agent/rag_system_update/rag_evaluation_research_papers/process_doc_and_query.py
```

Key files in the evaluation pack include:

- `process_doc_and_query.py` — main orchestration, source indexing, retrieval, evaluation loop, JSON output
- `evaluate_retrieval.py` — score one result set against an expected source paper
- `eval_cases.json` — the benchmark questions and expected source documents
- `eval_results.json` — generated result summary for the latest run

## Research workflow

The broader project still follows the same high-level pattern:

1. classify whether a request is in scope
2. plan a research path when needed
3. select tools and external evidence sources
4. index and query uploaded documents using the RAG pipeline
5. synthesize a final answer with citations and source grounding

## Notes

- The project keeps a separation between document parsing and retrieval logic.
- The document loader is intentionally strict about file type support.
- The PDF handler is the only file-type-specific branch in the current update structure.
- The older `rag_system` package remains as a legacy reference and should not be treated as the active implementation unless the project intentionally chooses to revert.
