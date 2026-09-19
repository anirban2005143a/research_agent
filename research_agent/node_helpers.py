"""Reusable helper functions for research graph nodes."""

import json
import re
from typing import Any
from urllib.parse import quote

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from .parsers import SourceMerge, llm_response_fixing_parser
from .prompts import (
    SOURCE_MERGE_INPUT_TEMPLATE,
    SOURCE_MERGE_SYSTEM_PROMPT,
    DRAFT_RESPONSE_INPUT_TEMPLATE,
    DRAFT_RESPONSE_FALLBACK_SYSTEM_PROMPT,
    DRAFT_RESPONSE_SYSTEM_PROMPT,
    MESSAGE_SUMMARY_SYSTEM_PROMPT,
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


def generate_draft_response(
    llm: Any,
    state: dict[str, Any],
    content_blocks: list[str],
    session_context: dict[str, Any],
) -> str:
    """Generate a standalone Markdown answer from the gathered research content."""
    research_material = "\n\n--- RESEARCH MATERIAL ---\n".join(
        f"Content block {index}: {content.strip()}"
        for index, content in enumerate(content_blocks, start=1)
        if str(content).strip()
    ) or "No research material was found for this query."
    memory = session_context or {}
    message_summary = state.get("message_summary", "")
    evaluation = state.get("evaluation") or "No prior evaluation is available."
    recent_context = state.get("messages", [])
    input_message = DRAFT_RESPONSE_INPUT_TEMPLATE.format(
        query=state["query"],
        evaluation=evaluation,
        user_info=memory.get("user_info", []),
        session_context=memory.get("session_context", []),
        message_summary=message_summary,
        recent_conversation=format_recent_messages(recent_context),
        research_material=research_material,
    )
    answer = invoke_llm(
        llm,
        "draft_llm",
        [
            SystemMessage(content=DRAFT_RESPONSE_SYSTEM_PROMPT),
            HumanMessage(content=input_message),
        ],
    )
    return str(getattr(answer, "content", answer)).strip()


def merge_sources(
    llm: Any,
    old_sources: list[str],
    recent_sources: list[str],
    draft_response: str,
) -> list[str]:
    """Merge and deduplicate the source list down to the five most relevant sources."""
    allowed_sources = {
        str(source).strip()
        for source in old_sources + recent_sources
        if str(source).strip()
    }
    parser = llm_response_fixing_parser(SourceMerge, llm)
    input_message = SOURCE_MERGE_INPUT_TEMPLATE.format(
        draft=draft_response,
        old_sources="\n".join(str(source).strip() for source in old_sources if str(source).strip()) or "None",
        recent_sources="\n".join(str(source).strip() for source in recent_sources if str(source).strip()) or "None",
    )
    result = parser.parse(
        invoke_llm(
            llm,
            "source_merge_llm",
            [
                SystemMessage(content=SOURCE_MERGE_SYSTEM_PROMPT),
                HumanMessage(content=f"{input_message}\n\n{parser.get_format_instructions()}"),
            ],
        ).content
    )
    return list(
        dict.fromkeys(
            str(source).strip()
            for source in result.sources
            if str(source).strip() in allowed_sources
        )
    )[:5]


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
