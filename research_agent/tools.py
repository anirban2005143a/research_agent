import time
from typing import Any

from ddgs import DDGS
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .config import settings
from .observability import log, retry_call, timed
from .rag_system import DocumentHandler, HybridRAG


class QueryInput(BaseModel):
    query: str = Field(description="The focused research question or retrieval query.")


class FileInput(BaseModel):
    source_name: str = Field(description="Exact uploaded or stored file name.")
    query: str = Field(
        default="",
        description="Optional question used to select the most relevant excerpts.",
    )


class QualityInput(BaseModel):
    source_text: str = Field(description="Source text or metadata to assess.")


def _web_search(query: str) -> str:
    log(f"TOOL web_search | query={query!r}")
    with timed("web_search"):
        results = retry_call(lambda: DDGS().text(query, max_results=5), "tool.web_search")
    if not results:
        return "No web sources found. The available external knowledge may not cover this query."
    return "\n\n".join(
        f"Title: {item.get('title', '')}\nURL: {item.get('href', '')}\nSnippet: {item.get('body', '')}"
        for item in results
    )


def _source_quality_check(source_text: str) -> str:
    markers = [
        "doi",
        "journal",
        "publisher",
        "government",
        "university",
        "arxiv",
        "methods",
    ]
    hits = sum(marker in source_text.lower() for marker in markers)
    return f"Quality signal: {hits}/{len(markers)} credibility markers found. Verify primary sources before strong claims."


def build_research_tools(rag: HybridRAG, document_handler: DocumentHandler | None = None) -> list[Any]:
    """Build the research tool set for the current session with clear, purpose-driven tool descriptions."""
    document_handler = document_handler or rag.document_handler

    @tool("web_search", args_schema=QueryInput)
    def web_search(query: str) -> str:
        """Use for current facts, recent events, broad background, policy updates, or general evidence that is not in the stored documents. Prefer precise, question-specific queries."""
        return _web_search(query)

    @tool("rag_search", args_schema=QueryInput)
    def rag_search(query: str) -> str:
        """Use when the question likely depends on uploaded or stored local documents. Search the indexed document corpus for matching evidence and return the strongest chunks with citations."""
        log(f"TOOL rag_search | query={query!r}")
        started = time.perf_counter()
        matches = retry_call(
            lambda: rag.retrieve(query, k=getattr(settings, "rag_top_k", 8)),
            "tool.rag_search",
        )
        log(f"TOOL rag_search | matches={len(matches)} | elapsed={time.perf_counter() - started:.2f}s")
        if not matches:
            return "No matching uploaded-document evidence was found. State this limitation explicitly."
        return "\n\n".join(
            f"Citation: {item['citation']}\nSource: {item['source']}\nScore: {item['score']}\nContent: {item['content']}"
            for item in matches
        )

    @tool("read_stored_file", args_schema=FileInput)
    def read_stored_file(source_name: str, query: str = "") -> str:
        """Use after a file name is known and you need the actual text of a specific uploaded document, usually to verify a claim or inspect a relevant section. Do not use this for vague file discovery."""
        log(f"TOOL read_stored_file | source={source_name!r} | query={query!r}")
        with timed("read_stored_file"):
            return retry_call(lambda: document_handler.read_stored_file(source_name, query=query), "tool.read_stored_file")

    @tool("list_stored_files")
    def list_stored_files() -> str:
        """Use when you need to discover which local files are available before choosing a document-specific read/search. Returns the stored file names and counts."""
        log("TOOL list_stored_files")
        files = document_handler.list_files()
        return (
            "\n".join(
                f"File: {item['source']} | type: {item.get('file_type', 'unknown')} | "
                f"pages: {item.get('pages', '?')} | chunks: {item['chunk_count']} | "
                f"metadata: {item.get('mime_type', 'unknown')}"
                for item in files
            )
            or "No documents are currently stored."
        )

    @tool("source_quality_check", args_schema=QualityInput)
    def source_quality_check(source_text: str) -> str:
        """Use to assess whether a source shows signs of authority (publisher, methods, DOI, institution, etc.). This does not prove correctness; it only flags credibility signals."""
        log(f"TOOL source_quality_check | chars={len(source_text)}")
        return _source_quality_check(source_text)

    wikipedia_backend = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=3))
    arxiv_backend = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=5, load_max_docs=5))

    @tool("wikipedia_search", args_schema=QueryInput)
    def wikipedia_search(query: str) -> str:
        """Use for concise background, definitions, historical context, or neutral overview material. Prefer more authoritative sources for technical or disputed claims."""
        log(f"TOOL wikipedia_search | query={query!r}")
        with timed("wikipedia_search"):
            return str(retry_call(lambda: wikipedia_backend.invoke(query), "tool.wikipedia_search"))

    @tool("arxiv_search", args_schema=QueryInput)
    def arxiv_search(query: str) -> str:
        """Use for academic papers, system design details, implementation behavior, algorithms, and technical literature. Best for research questions that require formal or peer-reviewed technical evidence."""
        log(f"TOOL arxiv_search | query={query!r}")
        with timed("arxiv_search"):
            return str(retry_call(lambda: arxiv_backend.invoke(query), "tool.arxiv_search"))

    return [
        web_search,
        rag_search,
        read_stored_file,
        list_stored_files,
        wikipedia_search,
        arxiv_search,
        source_quality_check,
    ]
