import json
from typing import Any

import wikipedia
from langchain_community.tools import DuckDuckGoSearchResults, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .config import settings
from .utils import log, retry_call
from .rag_system import DocumentHandler, HybridRAG
from .rag_system.data_types import SearchResults


class QueryInput(BaseModel):
    query: str = Field(description="The focused research question or retrieval query.")


class FileInput(BaseModel):
    source_name: str = Field(description="Exact uploaded or stored file name.")
    query: str = Field(
        default="",
        description="Optional question used to select the most relevant excerpts.",
    )


def _web_search(query: str) -> str:
    log(f"tool.web_search.started | query={query!r}")
    search_tool = DuckDuckGoSearchResults(
        num_results=5,
        output_format="json",
        keys_to_include=["title", "link", "snippet"],
    )
    results = retry_call(lambda: search_tool.invoke(query), "tool.web_search")
    if not results:
        return "No web sources found. The available external knowledge may not cover this query."
    return str(results)


def _arxiv_search(query: str, max_results: int = 5) -> str:
    """Search Arxiv using the current Client API instead of the legacy wrapper API."""
    import arxiv

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results = arxiv.Client().results(search)
    summaries = [
        (
            f"Published: {result.updated.date()}\n"
            f"Title: {result.title}\n"
            f"Authors: {', '.join(author.name for author in result.authors)}\n"
            f"URL: {result.entry_id}\n"
            f"Summary: {result.summary}"
        )
        for result in results
    ]
    return "\n\n".join(summaries) or "No good Arxiv result was found."


def build_research_tools(rag: HybridRAG, document_handler: DocumentHandler | None = None) -> list[Any]:
    """Build the research tool set for the current session with clear, purpose-driven tool descriptions."""
    document_handler = document_handler or rag.document_handler

    @tool("web_search", args_schema=QueryInput)
    def web_search(query: str) -> str:
        """Use for current facts, recent events, broad background, policy updates, or general evidence that is not in the stored documents. Prefer precise, question-specific queries."""
        return _web_search(query)

    @tool("rag_search", args_schema=QueryInput)
    def rag_search(query: str) -> dict[str, Any]:
        """Search the user's uploaded or stored documents using hybrid retrieval. Use this when the request refers to local files, uploaded evidence, or a topic that may be covered by the session's stored documents. It searches document contents and returns matching evidence with source and page citations. Do not use it for general web-only research when no relevant stored documents exist."""
        log(f"tool.rag_search.started | query={query!r}")
        matches = retry_call(
            lambda: rag.retrieve(query, k=getattr(settings, "rag_top_k", 8)),
            "tool.rag_search",
        )
        result_records = _rag_search_results_to_records(matches)
        log(f"tool.completed | name=rag_search | matches={len(result_records)}")
        return {
            "result_type": "rag_search_results",
            "results": result_records,
            "message": "Matching uploaded-document evidence found."
            if result_records
            else "No matching uploaded-document evidence was found.",
        }

    @tool("read_stored_file", args_schema=FileInput)
    def read_stored_file(source_name: str, query: str = "") -> str:
        """Read one specific document uploaded by the user from the session document store. Use this when the user names a file or asks to summarize, inspect, verify, or extract information from a particular uploaded document. Do not use it for web or Wikipedia content, and do not use it when no exact stored filename is known."""
        log(f"tool.read_stored_file.started | source={source_name!r} | query={query!r}")
        return retry_call(lambda: document_handler.read_stored_file(source_name), "tool.read_stored_file")

    @tool("list_stored_files")
    def list_stored_files() -> list[str]:
        """List the files uploaded by the user and stored in the current research session. Use this before read_stored_file when the user refers to an uploaded document without giving its exact filename, or asks what documents are available. This does not search document contents."""
        log("tool.list_stored_files.started")
        return retry_call(document_handler.list_files, "tool.list_stored_files")

    wikipedia.set_user_agent("ResearchAgentApp/1.0 (anirban@example.com)")
    wikipedia_tool = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(
            top_k_results=3,
            doc_content_chars_max=4000,
        )
    )

    @tool("wikipedia_search", args_schema=QueryInput)
    def wikipedia_search(query: str) -> str:
        """Use for concise background, definitions, historical context, or neutral overview material. Prefer more authoritative sources for technical or disputed claims."""
        log(f"tool.wikipedia_search.started | query={query!r}")
        try:
            try:
                result = wikipedia_tool.invoke(query)
            except json.JSONDecodeError as exc:
                log(f"tool.wikipedia_search.unavailable | error={exc!r}")
                return "Wikipedia search unavailable: the upstream response was not valid JSON."
            except Exception:
                result = retry_call(
                    lambda: wikipedia_tool.invoke(query), "tool.wikipedia_search"
                )
        except Exception as exc:
            log(f"tool.wikipedia_search.unavailable | error={exc!r}")
            return f"Wikipedia search unavailable for this query: {exc}"
        return str(result) or "Wikipedia returned no results for this query."

    @tool("arxiv_search", args_schema=QueryInput)
    def arxiv_search(query: str) -> str:
        """Use for academic papers, system design details, implementation behavior, algorithms, and technical literature. Best for research questions that require formal or peer-reviewed technical evidence."""
        log(f"tool.arxiv_search.started | query={query!r}")
        return retry_call(lambda: _arxiv_search(query), "tool.arxiv_search")

    return [
        web_search,
        rag_search,
        read_stored_file,
        list_stored_files,
        wikipedia_search,
        arxiv_search,
    ]


def _rag_search_results_to_records(search_results: SearchResults | Any) -> list[dict[str, Any]]:
    """Convert retrieval results into LangChain-safe JSON records."""
    if isinstance(search_results, SearchResults):
        results = search_results.results
    elif isinstance(search_results, dict):
        results = search_results.get("results", [])
    elif isinstance(search_results, list):
        results = search_results
    else:
        results = getattr(search_results, "results", None)
        if results is None:
            results = getattr(search_results, "items", None)
        if results is None:
            results = getattr(search_results, "matches", None)
        if results is None:
            log(
                "tool.rag_search.unrecognized_result | "
                f"type={type(search_results).__name__}"
            )
            return []

    records = []
    for result in results:
        document = getattr(result, "document", None)
        if document is None:
            document = result.get("document") if isinstance(result, dict) else None
        if document is None:
            continue
        metadata = document.metadata
        if isinstance(result, dict):
            chunk_id = result.get("chunk_id", "")
            score = result.get("final_score", 0.0)
        else:
            chunk_id = getattr(result, "chunk_id", "")
            score = getattr(result, "final_score", 0.0)
        records.append(
            {
                "chunk_id": chunk_id,
                "source": metadata.get("source", "unknown"),
                "citation": _citation_for(metadata),
                "content": document.page_content,
                "score": score,
                "metadata": metadata,
            }
        )
    return records


def _citation_for(metadata: dict[str, Any]) -> str:
    source = metadata.get("source", "unknown")
    page = metadata.get("page") or metadata.get("page_number")
    return f"{source}, page {page}" if page is not None else str(source)
