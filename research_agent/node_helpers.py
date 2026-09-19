"""Reusable helper functions for research graph nodes."""

import json
import re
from typing import Any
from urllib.parse import quote

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from .parsers import CitationMerge, ResearchDraft, llm_response_fixing_parser
from .prompts import (
    CITATION_MERGE_INPUT_TEMPLATE,
    CITATION_MERGE_SYSTEM_PROMPT,
    DRAFT_RESPONSE_INPUT_TEMPLATE,
    DRAFT_RESPONSE_FALLBACK_SYSTEM_PROMPT,
    DRAFT_RESPONSE_SYSTEM_PROMPT,
    MESSAGE_SUMMARY_SYSTEM_PROMPT,
    RAG_EVIDENCE_CONTEXT,
)
from .utils import log, retry_call


def normalize_tool_arguments(tool_name: str, arguments: dict) -> dict:
    """Keep LLM-generated tool arguments compatible with the public schemas."""
    arguments = dict(arguments or {})
    if tool_name == "read_stored_file" and "source_name" not in arguments:
        arguments["source_name"] = arguments.get("file_name", "")
    if tool_name in {"web_search", "rag_search", "wikipedia_search", "arxiv_search"}:
        return {"query": str(arguments.get("query", "")).strip()}
    if tool_name == "read_stored_file":
        return {
            "source_name": str(arguments.get("source_name", "")).strip(),
            "query": str(arguments.get("query", "")).strip(),
        }
    if tool_name == "list_stored_files":
        return {}
    return arguments


def append_recent_conversation_turn(
    state: dict[str, Any], query: str, response: str, llm: Any
) -> dict[str, Any]:
    """Append one completed turn while retaining five complete recent turns."""
    current_messages = list(state.get("messages", []))
    summary = state.get("message_summary", "").strip()
    new_turn = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": response},
    ]

    if len(current_messages) >= 10:
        removed = current_messages[:2]
        remaining = current_messages[2:]
        removed_text = format_recent_messages(removed)
        summary_input = (
            f"Existing older summary:\n{summary or 'None'}\n\n"
            f"Newly archived conversation turn:\n{removed_text}"
        )
        if removed_text and llm:
            try:
                summary = str(
                    invoke_llm(
                        llm,
                        "message_summary_llm",
                        [
                            SystemMessage(content=MESSAGE_SUMMARY_SYSTEM_PROMPT),
                            HumanMessage(content=summary_input),
                        ],
                    ).content
                ).strip()
            except Exception as exc:
                log(f"graph.message_summary.fallback | error={exc!r}")
                summary = "\n".join(
                    part for part in [summary, removed_text] if part
                ).strip()
        elif removed_text:
            summary = "\n".join(
                part for part in [summary, removed_text] if part
            ).strip()
    else:
        remaining = current_messages

    return {
        **state,
        "messages": remaining + new_turn,
        "message_summary": summary,
    }


def format_recent_messages(messages: list[Any]) -> str:
    """Format stored conversation turns as readable context for an LLM input prompt."""
    formatted_messages = []
    for message in messages:
        if isinstance(message, dict):
            role = message.get("role", "message")
            content = message.get("content", "")
        else:
            role = getattr(message, "type", "message")
            content = getattr(message, "content", str(message))
        if content:
            formatted_messages.append(f"{role}: {content}")
    return "\n".join(formatted_messages) or "No previous conversation is available."


def create_draft_and_select_citations(
    llm: Any,
    state: dict[str, Any],
    sources: list[dict[str, Any]],
    session_context: dict[str, Any],
) -> tuple[str, list[str]]:
    """Generate an evidence-grounded draft and return the citation strings it selected."""
    evidence_blocks = []
    for item in sources:
        evidence_blocks.append(
            f"Source: {item.get('source', '')}\n"
            f"Content: {item.get('content', '')}"
        )
    context = "\n\n--- EVIDENCE ---\n".join(evidence_blocks)
    memory = session_context
    message_summary = state.get("message_summary", "")
    recent_context = state.get("messages", [])
    input_message = (
        f"{RAG_EVIDENCE_CONTEXT}\n\n"
        f"{DRAFT_RESPONSE_INPUT_TEMPLATE.format(
            query=state['query'],
            user_info=memory.get('user_info', []),
            session_context=memory.get('session_context', []),
            message_summary=message_summary,
            recent_conversation=format_recent_messages(recent_context),
            evidence=context or 'No external evidence was found.',
        )}"
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
        answer = result.answer.strip()
        citations = list(dict.fromkeys(result.citations))[:3]
    except Exception as exc:
        log(f"graph.collect_informations.draft_fallback | error={exc!r}")
        answer = str(
            invoke_llm(
                llm,
                "draft_response_llm_fallback",
                [
                    SystemMessage(content=DRAFT_RESPONSE_FALLBACK_SYSTEM_PROMPT),
                    HumanMessage(
                        content=(
                            f"Question and research context:\n{input_message}\n\n"
                            "Write the actual answer now."
                        )
                    ),
                ],
            ).content
        )
        citations = list(
            dict.fromkeys(
                str(item.get("source", "")).strip()
                for item in sources
                if str(item.get("source", "")).strip()
            )
        )[:5]
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
        return list(dict.fromkeys(result.citations))[:5]
    except Exception as exc:
        log(f"graph.collect_informations.citation_merge_fallback | error={exc!r}")
        return list(dict.fromkeys(old_citations + recent_citations))[:5]


def tool_content_to_records(raw_content: Any, source_hint: str) -> list[dict[str, str]]:
    """Convert one successful tool result into source/content records."""
    records: list[dict[str, Any]] = []
    content = str(raw_content)
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and payload.get("result_type") == "rag_search_results":
        for result in payload.get("results", []):
            records.append(
                {
                    "content": str(result.get("content", "")),
                    "source": result.get("citation") or result.get("source", source_hint),
                }
            )
        return records
    if isinstance(payload, dict) and payload.get("result_type") == "wikipedia_search_results":
        for result in payload.get("results", []):
            records.append(
                {
                    "content": str(result.get("content", "")),
                    "source": result.get("source") or result.get("url") or source_hint,
                }
            )
        return records
    if isinstance(payload, list) and source_hint == "web_search":
        for result in payload:
            records.append(
                {
                    "content": str(result.get("snippet") or result.get("body", "")),
                    "source": result.get("link") or result.get("href") or result.get("title", source_hint),
                }
            )
        return records
    if isinstance(payload, str):
        content = payload
    elif payload is not None:
        content = json.dumps(payload, ensure_ascii=True)
    urls = re.findall(r"https?://[^\s)]+", content)
    source = urls[0].rstrip(".,") if urls else source_hint
    if not urls and source_hint == "wikipedia_search":
        page_match = re.search(r"^Page:\s*(.+)$", content, re.MULTILINE)
        if page_match:
            source = f"https://en.wikipedia.org/wiki/{quote(page_match.group(1).strip().replace(' ', '_'))}"
    records.append({"content": content, "source": source})
    return records


def invoke_llm(llm: Any, label: str, messages: list[BaseMessage], model: Any = None) -> Any:
    """Invoke an LLM with the shared retry policy and completion logging."""
    response = retry_call(lambda: (model or llm).invoke(messages), label)
    log(f"llm.completed | name={label} | response_chars={len(str(getattr(response, 'content', response)))}")
    return response
