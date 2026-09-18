from typing import Any, TypedDict

class ResearchState(TypedDict, total=False):
    query: str
    messages: list[Any]
    memory_context: dict[str, Any]
    scope_category: str
    tasks: list[str]
    current_task_index: int
    tool_responses: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    draft_response: str
    evaluation: dict[str, Any]
    iterations: int
    final_response: str
    hitl_question: str
    hitl_answer: str
    needs_hitl: bool
