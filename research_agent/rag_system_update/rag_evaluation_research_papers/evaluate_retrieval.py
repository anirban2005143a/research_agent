import argparse
import json
import os
import time
from pathlib import Path

from rag_adapter import normalize_results, retrieve


SCRIPT_DIR = Path(__file__).resolve().parent


def source_match(metadata, expected):
    vals = []
    for value in metadata.values():
        if value is not None:
            vals.append(str(value).lower())
    blob = " ".join(vals)
    exp = os.path.basename(expected).lower()
    return exp in blob or os.path.splitext(exp)[0] in blob or any(marker in blob for marker in ["/" + exp, "\\" + exp])


def page_match(metadata, pages):
    if not pages:
        return True
    for key in ("page", "page_number", "page_num"):
        if key in metadata:
            try:
                return int(metadata[key]) in pages or int(metadata[key]) + 1 in pages
            except Exception:
                pass
    return True


def rank_for_source(results, expected):
    for index, result in enumerate(results, start=1):
        if source_match(result.get("metadata", {}), expected):
            return index
    return None


def rr(rank, k):
    return 1.0 / rank if rank and rank <= k else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(SCRIPT_DIR / "eval_cases.json"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mode", default="hybrid", help="vector_only, bm25, hybrid, etc.")
    ap.add_argument("--out", default=str(SCRIPT_DIR / "eval_results.json"))
    args = ap.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.is_absolute():
        cases_path = (SCRIPT_DIR / cases_path).resolve()
    with cases_path.open("r", encoding="utf-8") as handle:
        cases = json.load(handle)

    rows = []
    for case in cases:
        t = time.perf_counter()
        try:
            results = normalize_results(retrieve(case["question"], k=args.k, mode=args.mode))
            err = None
        except Exception as exc:  # pragma: no cover - evaluation harness
            results = []
            err = repr(exc)
        latency = time.perf_counter() - t
        rank = rank_for_source(results, case["expected_source"])
        hit = rank is not None and rank <= args.k
        rows.append(
            {
                **{key: case[key] for key in ["id", "question", "expected_source", "relevant_pages"]},
                "rank": rank,
                "hit@k": hit,
                "rr@k": rr(rank, args.k),
                "latency_sec": latency,
                "results": [
                    {"metadata": result.get("metadata", {}), "content": result.get("content", "")[:500]}
                    for result in results
                ],
                "error": err,
            }
        )

    hits = sum(row["hit@k"] for row in rows)
    mrr = sum(row["rr@k"] for row in rows) / len(rows)
    avg_lat = sum(row["latency_sec"] for row in rows) / len(rows)
    summary = {
        "num_cases": len(rows),
        "k": args.k,
        "mode": args.mode,
        "Recall@K": hits / len(rows),
        "MRR@K": mrr,
        "avg_latency_sec": avg_lat,
        "errors": sum(bool(row["error"]) for row in rows),
    }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (SCRIPT_DIR / out_path).resolve()
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "cases": rows}, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
