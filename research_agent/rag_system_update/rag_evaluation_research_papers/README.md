# Research-paper RAG Evaluation Pack

This folder evaluates the active update-package RAG against a fixed set of research PDFs stored in the project.

## What this folder does

It measures how well the live retrieval pipeline can answer research questions by:

1. locating the PDF source directory
2. indexing all PDFs through `HybridRAG`
3. loading the benchmark questions from `eval_cases.json`
4. running each question against the indexed corpus
5. checking whether the expected paper is retrieved in the top-k results
6. writing the final summary and per-case scores to `eval_results.json`

This is a retrieval benchmark for the active project implementation, not a standalone demo or a mock system.

## Important files
- `process_doc_and_query.py`: main orchestration file for source processing, retrieval, batch evaluation, and JSON output
- `evaluate_retrieval.py`: evaluates a single retrieval result set against one expected source paper
- `eval_cases.json`: benchmark questions and the expected PDF for each case
- `eval_cases.jsonl`: same benchmark content in JSONL format for alternate tooling
- `eval_results.json`: output file containing the latest evaluation report

## Source directory
The evaluation system reads PDFs from:

```text
research_agent/rag_system_update/Artificial_Intelligence/
```

The code intentionally targets this directory and indexes all `.pdf` files it contains. It does not run over arbitrary folders.

## Execution flow

### Step 1: source processing
`process_doc_and_query.py` calls `process_source_directory()` in the main entrypoint.
This function loops over every PDF in the source directory and calls `HybridRAG.store_document()`.

### Step 2: query execution
For each case in `eval_cases.json`, the script reads the question and sends it through the live `HybridRAG` retriever.

### Step 3: single-result scoring
`evaluate_retrieval.py` checks whether the expected source paper appears in the returned result list and computes:

- `rank`
- `hit@k`
- `rr@k`

### Step 4: batch summary
The evaluation script loops all benchmark rows and computes aggregate values such as:

- `num_cases`
- `k`
- `mode`
- `Recall@K`
- `MRR@K`
- `avg_latency_sec`
- `errors`

## How to run it

From the project root:

```bash
python research_agent/rag_system_update/rag_evaluation_research_papers/process_doc_and_query.py
```

This runs the full benchmark using the active RAG engine and writes the output into `eval_results.json` in the same folder.

## Output
The script writes a JSON report with both:

- the overall summary metrics
- the per-case result rows including the question, expected source, rank, hit flag, rr score, and the retrieved result snippets

This output is meant to be reviewed after each run to validate whether the retrieval quality is acceptable or whether the evaluation setup itself needs adjustment.

## Improvement scope for the evaluation system

The current benchmark is intentionally a baseline, and the evaluation should evolve toward stronger checks such as:

- whether the retrieved content actually overlaps with the expected paper section
- whether the question is answered by that retrieved content
- whether the result is the expected document for that exact question
- a stronger measure than only filename matching

These are the core quality checks that the evaluation design should eventually cover in order to judge retrieval quality more accurately than simple metadata name matching.

## Design rule
This evaluation package speaks only to the active `HybridRAG` engine. It should not bypass the engine and directly manipulate the low-level document loader classes.

The folder exists to validate the project’s real retrieval pipeline under a controlled benchmark, not to replace the application’s main document indexing flow.
