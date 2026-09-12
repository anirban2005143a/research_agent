from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class ResearchState(TypedDict, total=False):
    session_id: str
    query: str
    messages: Annotated[list[Any], add_messages]
    tool_rounds: int
    user_preferences: dict[str, str]
    memory_context: dict[str, Any]
    scope_allowed: bool
    scope_reason: str
    plan: list[str]
    current_step: int
    sources: list[dict[str, Any]]
    retrieved_context: list[dict[str, Any]]
    draft: str
    critique: str
    iterations: int
    final_answer: str
    hitl_question: str
    hitl_answer: str
    needs_hitl: bool
    error: str
