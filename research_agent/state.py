from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class ResearchState(TypedDict, total=False):
    query: str
    messages: Annotated[list[Any], add_messages]
    memory_context: dict[str, Any]
    scope_allowed: bool
    scope_category: str
    out_of_scope_response: str
    plan: list[str]
    tool_responses: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    draft: str
    evaluation: str
    iterations: int
    final_answer: str
    hitl_question: str
    hitl_answer: str
    needs_hitl: bool
