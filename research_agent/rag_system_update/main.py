"""Small command-line checker for the RAG pipeline."""

import argparse
import json
import sys
from pathlib import Path

from .document_handler import SUPPORTED_EXTENSIONS
from .rag_engine import HybridRAG


class HelpfulArgumentParser(argparse.ArgumentParser):
    """Show examples when the user misses required arguments."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        print("\n[RAG][USAGE] Examples:", file=sys.stderr)
        print(
            '  python -m research_agent.rag_system_update.main '
            '"what is this document about?" --input documents/',
            file=sys.stderr,
        )
        print(
            '  python -m research_agent.rag_system_update.main '
            '"summarize the risks" --input file1.pdf file2.docx '
            '--session-id demo --k 5',
            file=sys.stderr,
        )
        print(
            '  python -m research_agent.rag_system_update.main '
            '"compare the findings" --input D:/docs/folder '
            '--session-id team-a',
            file=sys.stderr,
        )
        print(
            "  python -m research_agent.rag_system_update.main",
            file=sys.stderr,
        )
        self.exit(2, f"{self.prog}: error: {message}\n")


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
                if path.is_file()
                and path.suffix.lower() in SUPPORTED_EXTENSIONS
            )

        else:
            raise FileNotFoundError(
                f"Input path does not exist: {input_path}"
            )

        for path in candidates:
            path = path.resolve()

            if path not in seen:
                seen.add(path)
                collected.append(path)

    return collected


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = HelpfulArgumentParser(
        description="Index documents and test research RAG retrieval."
    )

    parser.add_argument(
        "query",
        nargs="?",
        help=(
            "Question to retrieve evidence for. "
            "If omitted, you will be prompted for it."
        ),
    )

    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        action="append",
        help=(
            "One or more files or directories to index before querying. "
            "Example: --input file1.pdf file2.docx or --input docs/"
        ),
    )

    parser.add_argument(
        "--session-id",
        default="rag-cli",
        help="RAG collection namespace",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of final results",
    )

    return parser


def _prompt_for_query() -> str:
    """Prompt the user for a query."""

    while True:
        query = input("\n[RAG] Enter your question: ").strip()

        if query:
            return query

        print("[RAG] Question cannot be empty. Please try again.")


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
        invalid_paths = [
            path for path in input_paths if not path.expanduser().exists()
        ]

        if invalid_paths:
            print("\n[RAG] The following path(s) do not exist:")

            for path in invalid_paths:
                print(f"      - {path}")

            print("\n[RAG] Please enter the paths again.")
            continue

        return input_paths


def _get_input_paths(args: argparse.Namespace) -> list[Path]:
    """Get input paths from CLI arguments or interactive input."""

    if args.input:
        return [
            path
            for group in args.input
            for path in group
        ]

    return _prompt_for_input_paths()


def main() -> None:
    """Run the RAG command-line checker."""

    parser = build_parser()
    args = parser.parse_args()

    # ---------------------------------------------------------
    # Get query
    # ---------------------------------------------------------
    if args.query:
        query = args.query
    else:
        query = _prompt_for_query()

    # ---------------------------------------------------------
    # Get input files/directories
    # ---------------------------------------------------------
    input_paths = _get_input_paths(args)

    # ---------------------------------------------------------
    # Create RAG engine
    # ---------------------------------------------------------
    print("\n[RAG] Initializing RAG engine...")

    rag = HybridRAG(session_id=args.session_id)

    # ---------------------------------------------------------
    # Find documents
    # ---------------------------------------------------------
    if input_paths:
        print("\n[RAG] Searching for supported documents...")

        files = _find_files(input_paths)

        if not files:
            raise SystemExit(
                "\n[RAG] No supported documents found under the "
                "provided input paths:\n"
                + "\n".join(f"  - {path}" for path in input_paths)
            )

        print(f"[RAG] Found {len(files)} document(s).")

        # -----------------------------------------------------
        # Index documents
        # -----------------------------------------------------
        for index, path in enumerate(files, start=1):
            print(
                f"\n[RAG] Indexing document "
                f"{index}/{len(files)}: {path}"
            )

            source_name = path.name

            # If the file belongs to one of the supplied directories,
            # preserve its relative path as the source name.
            for directory in input_paths:
                directory = directory.expanduser()

                if directory.is_dir():
                    try:
                        relative_path = path.relative_to(
                            directory.resolve()
                        )
                    except ValueError:
                        continue

                    source_name = str(relative_path)
                    break

            rag.store_document(
                path,
                source_name=source_name,
            )

        print("\n[RAG] Document indexing complete.")

    # ---------------------------------------------------------
    # Retrieve results
    # ---------------------------------------------------------
    print(f"\n[RAG] Query: {query}")
    print("[RAG] Searching for relevant evidence...")

    results = rag.retrieve(
        query,
        k=max(1, args.k),
    )

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------
    print(f"\n[RAG][RESULTS] {len(results)} results")

    if not results:
        print("[RAG] No relevant results found.")
        return

    for index, result in enumerate(results, start=1):
        metadata = result["metadata"]
        score = metadata.get("score", 0.0)
        rrf_score = metadata.get("rrf_score", 0.0)
        cross_encoder_score = metadata.get("cross_encoder_score", 0.0)

        print(
            f"\n--- {index} | "
            f"score={score} | "
            f"RRF={rrf_score} | "
            f"CE={cross_encoder_score} ---"
        )
        print(f"File: {metadata.get('filename', '')}")
        print(f"Title: {metadata.get('title', '')}")

        if metadata.get("authors"):
            print(f"Authors: {metadata['authors']}")

        if metadata.get("section_heading"):
            print(f"Section: {metadata['section_heading']}")

        print("Content:")
        print(result["content"])

        print("Metadata:")
        print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
