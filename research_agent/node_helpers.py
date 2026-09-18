"""Reusable helper functions for research graph nodes."""

import json
import re
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from .parsers import CitationMerge, ResearchDraft, llm_response_fixing_parser
from .prompts import (
    CITATION_MERGE_INPUT_TEMPLATE,
    CITATION_MERGE_SYSTEM_PROMPT,
    DRAFT_RESPONSE_INPUT_TEMPLATE,
    DRAFT_RESPONSE_SYSTEM_PROMPT,
    RAG_EVIDENCE_CONTEXT,
)
from .utils import log, retry_call


def create_draft_and_select_citations(
    llm: Any,
    state: dict[str, Any],
    sources: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    """Generate an evidence-grounded draft and return the citation strings it selected."""
    evidence_blocks = []
    for item in sources:
        locations = item.get("locations", [])
        location = "; ".join(locations) or item.get("source", "")
        evidence_blocks.append(
            f"Source: {item.get('source', '')}\n"
            f"Location: {location}\n"
            f"Metadata: {json.dumps(item.get('metadata', {}), ensure_ascii=True)}\n"
            f"Content: {item.get('content', '')}"
        )
    context = "\n\n--- EVIDENCE ---\n".join(evidence_blocks)
    memory = state.get("memory_context", {})
    input_message = (
        f"{RAG_EVIDENCE_CONTEXT}\n\n"
        f"{DRAFT_RESPONSE_INPUT_TEMPLATE.format(query=state['query'], preferences=memory.get('preferences', {}), summary=memory.get('summary', ''), evidence=context or 'No external evidence was found.')}"
    )
    parser = llm_response_fixing_parser(ResearchDraft, llm)
    try:
        result = parser.parse(
            invoke_llm(
                llm,
                "draft_llm",
                [
                    SystemMessage(content=DRAFT_RESPONSE_SYSTEM_PROMPT),
                    HumanMessage(content=f"{input_message}\n\n{parser.get_format_instructions()}"),
                ],
            ).content
        )
        answer = result.answer
        citations = result.citations
    except Exception as exc:
        log(f"graph.collect_informations.draft_fallback | error={exc!r}")
        answer = str(
            invoke_llm(
                llm,
                "draft_response_llm_fallback",
                [SystemMessage(content=DRAFT_RESPONSE_SYSTEM_PROMPT), HumanMessage(content=input_message)],
            ).content
        )
        citations = []
    log(f"graph.collect_informations.completed | response_chars={len(str(answer))} | citation_count={len(citations)}")
    return str(answer), citations


def merge_citations(
    llm: Any,
    old_citations: list[str],
    recent_citations: list[str],
    draft_response: str,
) -> list[str]:
    """Ask the LLM to deduplicate citations and keep only those needed by the draft."""
    parser = llm_response_fixing_parser(CitationMerge, llm)
    input_message = CITATION_MERGE_INPUT_TEMPLATE.format(
        draft=draft_response,
        old_citations="\n".join(old_citations) or "None",
        recent_citations="\n".join(recent_citations) or "None",
    )
    try:
        result = parser.parse(
            invoke_llm(
                llm,
                "citation_merge_llm",
                [
                    SystemMessage(content=CITATION_MERGE_SYSTEM_PROMPT),
                    HumanMessage(content=f"{input_message}\n\n{parser.get_format_instructions()}"),
                ],
            ).content
        )
        return result.citations
    except Exception as exc:
        log(f"graph.collect_informations.citation_merge_fallback | error={exc!r}")
        return list(dict.fromkeys(old_citations + recent_citations))


def tool_messages_to_records(tool_messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Convert tool messages into raw source/content records for graph state."""
    records: list[dict[str, Any]] = []
    for message in tool_messages:
        raw_content = message.content
        content = str(raw_content)
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and payload.get("result_type") == "rag_search_results":
            for result in payload.get("results", []):
                records.append(
                    {
                        "source": result.get("source", message.name or "rag_search"),
                        "content": result.get("content", ""),
                        "locations": [result.get("citation", "")],
                        "metadata": result.get("metadata", {}),
                        "tool_call_id": message.tool_call_id,
                    }
                )
            continue
        records.append(
            {
                "source": message.name or "research tool",
                "content": payload if payload is not None else raw_content,
                "locations": re.findall(r"https?://[^\s)]+", content),
                "metadata": {
                    "tool": message.name or "research tool",
                    "tool_call_id": message.tool_call_id,
                },
                "tool_call_id": message.tool_call_id,
            }
        )
    return records


def invoke_llm(llm: Any, label: str, messages: list[BaseMessage], model: Any = None) -> Any:
    """Invoke an LLM with the shared retry policy and completion logging."""
    response = retry_call(lambda: (model or llm).invoke(messages), label)
    log(f"llm.completed | name={label} | response_chars={len(str(getattr(response, 'content', response)))}")
    return response
