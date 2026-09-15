"""Small command-line checker for the RAG pipeline."""

import argparse
import json
import sys
from pathlib import Path
from .document_handler import SUPPORTED_EXTENSIONS
from .rag_engine import HybridRAG


def _find_files(input_paths: list[Path] | None) -> list[Path]:
    """Find supported documents from files and directories."""

    if not input_paths:
        return []

    collected: list[Path] = []
    seen: set[Path] = set()

    for input_path in input_paths:
        input_path = input_path.expanduser()

        if input_path.is_file():
            candidates = [input_path]

        elif input_path.is_dir():
            candidates = sorted(
                path
                for path in input_path.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            )

        else:
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        for path in candidates:
            path = path.resolve()

            if path not in seen:
                seen.add(path)
                collected.append(path)

    return collected


def _prompt_for_query() -> str:
    """Prompt the user for a query."""

    while True:
        query = input("\n[RAG] Enter your question: ").strip()

        if query:
            return query

        print("[RAG] Question cannot be empty. Please try again.")


def _prompt_for_session_id() -> str:
    """Prompt the user for a query."""

    while True:
        query = input("\n[RAG] Enter your session ID: ").strip()

        if query:
            return query

        print("[RAG] session ID cannot be empty. Please try again.")


def _prompt_for_top_K() -> int:
    """Prompt the user for a query."""

    while True:
        query = input("\n[RAG] Enter your value of top K: ").strip()

        if query:
            return query

        print("[RAG] value of top K cannot be empty. Please try again.")


def _prompt_for_input_paths() -> list[Path]:
    """Prompt the user for document or directory paths."""

    print(
        "\n[RAG] Enter file/folder path(s) to index."
        "\n      Multiple paths can be separated by commas."
        "\n      Example: documents/, report.pdf, D:/research/papers"
    )

    while True:
        raw_paths = input("\n[RAG] Input path(s): ").strip()

        if not raw_paths:
            print("[RAG] Please provide at least one file or folder.")
            continue

        input_paths = [
            Path(path.strip().strip('"').strip("'"))
            for path in raw_paths.split(",")
            if path.strip()
        ]

        if not input_paths:
            print("[RAG] Please provide at least one valid path.")
            continue

        # Check paths before continuing.
        invalid_paths = [path for path in input_paths if not path.expanduser().exists()]

        if invalid_paths:
            print("\n[RAG] The following path(s) do not exist:")

            for path in invalid_paths:
                print(f"      - {path}")

            print("\n[RAG] Please enter the paths again.")
            continue

        return input_paths


def main() -> None:
    """Run the RAG command-line checker."""

    # ---------------------------------------------------------
    # Get query
    # ---------------------------------------------------------
    query = _prompt_for_query()

    # ---------------------------------------------------------
    # Get session ID
    # ---------------------------------------------------------
    session_id = _prompt_for_session_id()

    # ---------------------------------------------------------
    # Get K
    # ---------------------------------------------------------
    k = _prompt_for_top_K()

    print(type(k))

    # ---------------------------------------------------------
    # Get input files/directories
    # ---------------------------------------------------------
    # input_paths = _prompt_for_input_paths()

    # ---------------------------------------------------------
    # Create RAG engine
    # ---------------------------------------------------------
    print("\n[RAG] Initializing RAG engine...")

    rag = HybridRAG(session_id=session_id)

    # # ---------------------------------------------------------
    # # Find documents
    # # ---------------------------------------------------------
    # if input_paths:
    #     print("\n[RAG] Searching for supported documents...")

    #     files = _find_files(input_paths)

    #     if not files:
    #         raise SystemExit(
    #             "\n[RAG] No supported documents found under the "
    #             "provided input paths:\n"
    #             + "\n".join(f"  - {path}" for path in input_paths)
    #         )

    #     print(f"[RAG] Found {len(files)} document(s).")

    #     # -----------------------------------------------------
    #     # Index documents
    #     # -----------------------------------------------------
    #     for index, path in enumerate(files, start=1):
    #         print(
    #             f"\n[RAG] Indexing document "
    #             f"{index}/{len(files)}: {path}"
    #         )

    #         rag.store_document(path)

    #     print("\n[RAG] Document indexing complete.")

    # ---------------------------------------------------------
    # Retrieve results
    # ---------------------------------------------------------
    print(f"\n[RAG] Query: {query}")
    print("[RAG] Searching for relevant evidence...")

    results = rag.retrieve(
        query,
        k=max(1, int(k)),
    )

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------
    print(results.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
