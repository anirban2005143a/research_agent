"""Standalone RAG checker: index local files and inspect retrieved chunks."""

import argparse
import json
from pathlib import Path

from .document_handler import SUPPORTED_EXTENSIONS


def _files_to_index(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index documents and inspect HybridRAG retrieval results.")
    parser.add_argument("query", nargs="?", help="Query to run after indexing")
    parser.add_argument("--input", type=Path, help="A file or directory to index")
    parser.add_argument("--session-id", default="rag-cli", help="Persistent RAG namespace (default: rag-cli)")
    parser.add_argument("--k", type=int, default=8, help="Maximum number of chunks to print")
    parser.add_argument("--no-index", action="store_true", help="Query the existing session without indexing")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.query:
        raise SystemExit('Provide a query, for example: python -m research_agent.rag_system.main "what is ...?"')
    from .rag_engine import HybridRAG

    rag = HybridRAG(session_id=args.session_id)
    if args.input and not args.no_index:
        files = _files_to_index(args.input)
        if not files:
            raise SystemExit(f"No supported documents found under {args.input}")
        for path in files:
            source_name = str(path.relative_to(args.input)) if args.input.is_dir() else path.name
            count = rag.store_document(path, source_name=source_name)
            print(f"Indexed {count:>4} chunks | {source_name}")
    results = rag.retrieve(args.query, k=max(1, args.k))
    print(f"\nQuery: {args.query}\nMatches: {len(results)}")
    for index, result in enumerate(results, start=1):
        print(f"\n--- Match {index} | score={result['score']} | {result['citation']} ---")
        print(f"Chunk ID: {result['chunk_id']}")
        print("Metadata:")
        print(json.dumps(result["metadata"], indent=2, default=str))
        print("Content:")
        print(result["content"])


if __name__ == "__main__":
    main()