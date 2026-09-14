# Research-paper RAG Evaluation Pack

This folder evaluates the active update-package RAG against a fixed set of 10 AI research papers.

## What is included
- `eval_cases.json` / `eval_cases.jsonl`: 50 retrieval questions with expected source papers.
- `rag_adapter.py`: integration layer that indexes the actual PDF set and calls the current `HybridRAG` pipeline.
- `evaluate_retrieval.py`: computes Recall@K, MRR@K, average latency, and per-case results.

## How it works
The adapter automatically:
- finds the PDFs under `research_agent/rag_system_update/Artificial_Intelligence/`
- builds a reusable evaluation session through `HybridRAG`
- indexes those documents once
- runs the selected retrieval mode (`hybrid`, `vector_only`, or `bm25`)
- returns the chunk results in the format expected by the evaluator

This keeps the evaluation hooked to the real system instead of a placeholder adapter.

## Run it
From the project root:

```bash
python research_agent/rag_system_update/rag_evaluation_research_papers/evaluate_retrieval.py --k 5 --mode hybrid
```

Or from inside the evaluation folder:

```bash
python evaluate_retrieval.py --k 5 --mode hybrid
```

Ablation runs:

```bash
python research_agent/rag_system_update/rag_evaluation_research_papers/evaluate_retrieval.py --k 5 --mode vector_only --out results_vector.json
python research_agent/rag_system_update/rag_evaluation_research_papers/evaluate_retrieval.py --k 5 --mode bm25 --out results_bm25.json
python research_agent/rag_system_update/rag_evaluation_research_papers/evaluate_retrieval.py --k 5 --mode hybrid --out results_hybrid.json
```

## Evaluation goal
The primary metric is whether the correct paper appears in the top-k results. The script also reports MRR@K, latency, and a per-case error log so you can compare retrieval quality across search modes.

## Output
The script writes a JSON report to `eval_results.json` by default, or to the file you pass with `--out`.
