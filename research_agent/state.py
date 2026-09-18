from typing import Annotated, Any, TypedDict


class ResearchState(TypedDict, total=False):
    query: Annotated[str, "Clean research query used by every downstream node."]
    messages: Annotated[
        list[Any],
        "Only the most recent 5 user/assistant turns remain in graph state; older turns are summarized and moved out of state.",
    ]
    message_summary: Annotated[
        str,
        "Condensed summary of older turns that were trimmed from the graph state conversation window.",
    ]
    scope_category: Annotated[
        str, "LLM scope classification: out_of_scope or in_scope."
    ]
    tasks: Annotated[list[str], "Ordered research tasks created by the planning node."]
    current_task_index: Annotated[
        int, "Zero-based index of the task currently being executed."
    ]
    tool_responses: Annotated[
        list[dict[str, Any]],
        "Raw tool records containing content and source information.",
    ]
    citations: Annotated[
        list[str], "Unique citation strings selected and merged by the LLM."
    ]
    draft_response: Annotated[
        str, "Latest synthesized research response before evaluation."
    ]
    evaluation: Annotated[
        dict[str, Any],
        "Evaluation verdict and improvement scopes for the latest draft.",
    ]
    iterations: Annotated[int, "Number of draft evaluations completed in this run."]
    final_response: Annotated[
        str, "User-facing response produced at graph finalization."
    ]
    hitl_question: Annotated[
        str, "Clarification question currently waiting for a user answer."
    ]
    hitl_answer: Annotated[str, "User answer to the clarification question."]
    needs_hitl: Annotated[
        bool, "Whether graph execution is waiting for human clarification."
    ]
