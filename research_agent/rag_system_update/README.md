# Research Agent RAG System (Update Package)

This folder contains the active RAG implementation for the project. It is the refactored and cleaner version of the document ingestion and retrieval flow, and it is the one currently used by the project update path.

## What this package does

The package is responsible for:

- accepting uploaded files
- validating the allowed document extensions
- choosing the correct file-specific handler
- loading raw text from the file
- cleaning and normalizing content
- extracting basic metadata
- chunking the document content for indexing
- storing chunks in Chroma
- retrieving and ranking the relevant results

## Supported file types

The allowed document types are intentionally strict:

- `.pdf`
- `.md`
- `.ppt`
- `.pptx`
- `.docx`
- `.txt`

Any other extension is rejected by the base `DocumentHandler`.

## Package structure

```text
research_agent/rag_system_update/
├── __init__.py
├── README.md
├── main.py
├── rag_engine.py
├── data_models.py
├── dense_retriever.py
├── lexical_retriever.py
├── overall_ranker.py
├── rrf_ranker.py
├── cross_encoder_ranker.py
└── document_handler/
    ├── __init__.py
    ├── document_handler.py
    └── pdf_handler.py
```

## Document handler design

The current pattern is a parent/child handler structure:

- `DocumentHandler`: generic base logic
- `PDFDocumentHandler`: PDF-specific parsing and heading extraction

This keeps the generic path simple and keeps PDF-only behavior isolated so it does not pollute the base class logic.

## Base handler responsibilities

`DocumentHandler` is responsible for:

- storing uploaded files in the session directory
- validating file paths and extensions
- dispatching the appropriate handler for a file
- loading source text for supported non-PDF types
- cleaning and normalizing text
- generating chunks with the configured text splitter
- preparing a final list of chunked `Document` objects

## PDF handler responsibilities

`PDFDocumentHandler` adds the PDF-specific behavior:

- reading the PDF with `pypdf`
- opening the PDF with `pymupdf`
- extracting text page by page
- reading PDF metadata for title and authors
- extracting a heading using markdown, fallback text, and the PDF outline when available
- returning page-level documents before the general chunking step

## Main processing flow

```text
File path
  -> DocumentHandler.prepare_file()
  -> _get_document_handler()
  -> PDFDocumentHandler or base handler
  -> load_document()
  -> prepare_document()
  -> extract metadata and clean text
  -> generate_chunks()
  -> return chunk list
```

This is the actual flow used when a file is stored and indexed by the update RAG pipeline.

## Metadata and chunking

The update package keeps the metadata simple and useful for retrieval:

- filename
- source
- title
- heading
- section_heading
- page number
- authors (when available)

Chunk generation is done with the project text splitter settings (`RAG_CHUNK_SIZE` and `RAG_CHUNK_OVERLAP`), then the final chunk list is returned for vector storage.

## Orchestration layer

`rag_engine.py` is the central orchestration entry for this package. It:

- creates the session document storage directory
- saves uploaded files
- prepares chunked documents with the handler
- adds them to the vector store
- handles retrieval and ranking through the dedicated components

## Ranking pipeline

The update package uses a clear separation of ranking responsibilities:

- `dense_retriever.py`: vector similarity retrieval
- `lexical_retriever.py`: BM25 lexical retrieval
- `rrf_ranker.py`: reciprocal rank fusion
- `cross_encoder_ranker.py`: reranking via cross-encoder model
- `overall_ranker.py`: final score aggregation and top-k selection

## Running it

From the project root:

```powershell
python -m research_agent.rag_system_update.main
```

This exercises the active update-package document pipeline and retrieval flow.

## Important note

The `rag_system_update` package is the current refactored implementation. The older `rag_system` package remains as a separate legacy implementation and is not the active update path unless deliberately switched back.
