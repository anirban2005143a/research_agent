import json
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt
from .parsers import (
    ClarificationDecision,
    ResearchPlan,
    ResearchToolSelection,
    ResponseEvaluation,
    ScopeCategory,
    ScopeDecision,
    llm_response_fixing_parser,
)
from .prompts import (
    CLARIFY_QUERY_INPUT_TEMPLATE,
    CLARIFY_QUERY_SYSTEM_PROMPT,
    CONVERSATION_CONTEXT_TEMPLATE,
    EVALUATE_RESPONSE_SYSTEM_PROMPT,
    HITL_CLARIFICATION_QUESTION,
    OUT_OF_SCOPE_INPUT_TEMPLATE,
    OUT_OF_SCOPE_RESPONSE_SYSTEM_PROMPT,
    PLANNING_INPUT_TEMPLATE,
    PLANNING_SYSTEM_PROMPT,
    EXECUTE_TASK_INPUT_TEMPLATE,
    EXECUTE_TASK_SYSTEM_PROMPT,
    SCOPE_GATE_SYSTEM_PROMPT,
    SINGLE_QUERY_INPUT_TEMPLATE,
    UNCLEAR_QUERY_INPUT_TEMPLATE,
    UNCLEAR_QUERY_RESPONSE_SYSTEM_PROMPT,
    AVAILABLE_TOOLS_TEMPLATE,
)
from .node_helpers import (
    append_recent_state_message,
    create_draft_and_select_citations,
    format_recent_messages,
    invoke_llm,
    merge_citations,
    normalize_tool_arguments,
    tool_content_to_records,
)
from .utils import log, log_function


class ResearchNodes:
    @log_function
    def scope_gate(self, state):
        """Classify the request and route unrelated requests away from research."""
        query = str(state.get("query", "")).strip()
        parser = llm_response_fixing_parser(ScopeDecision, self.llm)
        input_message = f"User request:\n{query}"
        try:
            decision = parser.parse(
                invoke_llm(
                    self.llm,
                    "scope_gate_llm",
                    [
                        SystemMessage(content=SCOPE_GATE_SYSTEM_PROMPT),
                        HumanMessage(
                            content=f"{input_message}\n\n{parser.get_format_instructions()}"
                        ),
                    ],
                ).content
            )
            category = decision.category
            return {"scope_category": category.value}
        except Exception as exc:
            log(f"graph.scope_gate.fallback | action=research | error={exc!r}")
            return {"scope_category": ScopeCategory.IN_SCOPE.value}

    @log_function
    def out_of_scope_response(self, state):
        """Explain the research scope and guide an unrelated request toward a research question."""
        memory = self.session_memory.context()
        recent_conversation = format_recent_messages(state.get("messages", []))
        input_message = OUT_OF_SCOPE_INPUT_TEMPLATE.format(
            query=state.get("query", ""),
        ) + (
            f"\n\nRemembered user information: {memory.get('user_info', [])}"
            f"\nRemembered session context: {memory.get('session_context', [])}"
            f"\n\nOlder summarized context: {state.get('message_summary', '') or 'None'}"
            f"\n\nRecent conversation: {recent_conversation}"
        )
        response = invoke_llm(
            self.llm,
            "out_of_scope_response_llm",
            [
                SystemMessage(content=OUT_OF_SCOPE_RESPONSE_SYSTEM_PROMPT),
                HumanMessage(content=input_message),
            ],
        )
        return {
            "final_response": str(response.content).strip(),
            "citations": [],
            "needs_hitl": False,
        }

    @log_function
    def clarify_query(self, state):
        """Decide whether clarification is needed and store a clean final query."""
        parser = llm_response_fixing_parser(ClarificationDecision, self.llm)
        query = state.get("query", "")
        hitl_answer = str(state.get("hitl_answer", "")).strip()
        conversation = format_recent_messages(state.get("messages", []))
        if hitl_answer:
            input_message = CLARIFY_QUERY_INPUT_TEMPLATE.format(
                query=query, clarification_answer=hitl_answer
            )
            input_message = f"{input_message}\n\n{CONVERSATION_CONTEXT_TEMPLATE.format(conversation=conversation)}"
            try:
                decision = parser.parse(
                    invoke_llm(
                        self.llm,
                        "clarified_query_llm",
                        [
                            SystemMessage(content=CLARIFY_QUERY_SYSTEM_PROMPT),
                            HumanMessage(
                                content=f"{input_message}\n\n{parser.get_format_instructions()}"
                            ),
                        ],
                    ).content
                )
                final_query = decision.final_query.strip() or f"{query} {hitl_answer}"
            except Exception as exc:
                log(f"graph.clarify_query.final_query_fallback | error={exc!r}")
                final_query = f"{query} Additional context: {hitl_answer}"
            return {
                "query": final_query,
                "hitl_answer": "",
                "needs_hitl": False,
            }

        input_message = (
            f"{SINGLE_QUERY_INPUT_TEMPLATE.format(query=query)}\n\n"
            f"{CONVERSATION_CONTEXT_TEMPLATE.format(conversation=conversation)}"
        )
        try:
            decision = parser.parse(
                invoke_llm(
                    self.llm,
                    "clarify_query_llm",
                    [
                        SystemMessage(content=CLARIFY_QUERY_SYSTEM_PROMPT),
                        HumanMessage(
                            content=f"{input_message}\n\n{parser.get_format_instructions()}"
                        ),
                    ],
                ).content
            )
        except Exception as exc:
            log(f"graph.clarify_query.decision_fallback | error={exc!r}")
            decision = ClarificationDecision(
                needs_clarification=False,
                question="",
                final_query=query,
            )

        if not decision.needs_clarification:
            return {
                "query": decision.final_query.strip() or query,
                "needs_hitl": False,
            }

        question = decision.question.strip() or HITL_CLARIFICATION_QUESTION
        answer = interrupt({"kind": "research_scope", "question": question})
        answer_text = "" if answer is None else str(answer).strip()
        if answer_text:
            return {"hitl_answer": answer_text, "needs_hitl": False}

        memory = self.session_memory.context()
        unclear_input = (
            UNCLEAR_QUERY_INPUT_TEMPLATE.format(query=query)
            + f"\n\nUser information: {memory.get('user_info', [])}"
            + f"\nSession context: {memory.get('session_context', [])}"
            + f"\nOlder summarized context: {state.get('message_summary', '') or 'None'}"
            + f"\nRecent conversation: {conversation}"
        )
        response = invoke_llm(
            self.llm,
            "unclear_query_response_llm",
            [
                SystemMessage(content=UNCLEAR_QUERY_RESPONSE_SYSTEM_PROMPT),
                HumanMessage(content=unclear_input),
            ],
        )
        return {
            "final_response": str(response.content).strip(),
            "citations": [],
            "needs_hitl": False,
        }

    @log_function
    def plan(self, state):
        """Create the next evidence-gathering plan from the query and current evaluation."""
        parser = llm_response_fixing_parser(ResearchPlan, self.llm)
        memory = self.session_memory.context()
        message_summary = state.get("message_summary", "")
        input_message = PLANNING_INPUT_TEMPLATE.format(
            query=state["query"],
            clarification=state.get("hitl_answer", "none"),
            draft=state.get("draft_response", "No draft exists yet."),
            evaluation=state.get("evaluation", "No evaluation exists yet."),
            user_info=memory.get("user_info", []),
            session_context=memory.get("session_context", []),
            message_summary=message_summary,
            recent_conversation=format_recent_messages(state.get("messages", [])),
        )
        try:
            result = invoke_llm(
                self.llm,
                "planner_llm",
                [
                    SystemMessage(content=PLANNING_SYSTEM_PROMPT),
                    HumanMessage(
                        content=f"{input_message}\n\n{parser.get_format_instructions()}"
                    ),
                ],
            )
            plan = parser.parse(result.content)
            tasks = [task.strip() for task in plan.tasks if task and task.strip()][:5]
            if not tasks:
                tasks = [state["query"]]
            log(f"graph.plan.created | task_count={len(tasks)}")
            for index, task in enumerate(tasks, start=1):
                log(f"graph.plan.task | index={index} | task={task!r}")
            return {"tasks": tasks, "current_task_index": 0, "tool_messages": []}
        except Exception as exc:
            log(f"graph.plan.fallback | action=original_query | error={exc!r}")
            return {
                "tasks": [state["query"]],
                "current_task_index": 0,
                "tool_messages": [],
            }

    @log_function
    def execute_task(self, state):
        """Select and run the tools needed for the current research task."""
        tasks = state.get("tasks", [state["query"]])
        task_index = state.get("current_task_index", 0)
        task = tasks[task_index] if task_index < len(tasks) else state["query"]
        memory = self.session_memory.context()
        message_summary = state.get("message_summary", "")
        input_message = EXECUTE_TASK_INPUT_TEMPLATE.format(
            query=state["query"],
            task=task,
            user_info=memory.get("user_info", []),
            session_context=memory.get("session_context", []),
            message_summary=message_summary,
            recent_conversation=format_recent_messages(state.get("messages", [])),
        )
        available_tools = "\n".join(
            f"- {tool.name}: {tool.description or 'No description provided.'}"
            for tool in self.tools
        )
        parser = llm_response_fixing_parser(ResearchToolSelection, self.llm)
        messages = [
            SystemMessage(content=EXECUTE_TASK_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"{input_message}\n\n"
                    f"{AVAILABLE_TOOLS_TEMPLATE.format(tools=available_tools)}\n\n"
                    f"{parser.get_format_instructions()}"
                )
            ),
        ]
        messages.extend(
            message
            for message in state.get("messages", [])
            if isinstance(message, (HumanMessage, SystemMessage, AIMessage))
        )
        try:
            selection_response = invoke_llm(self.llm, "execute_task_llm", messages)
            raw_selection = getattr(selection_response, "content", selection_response)
            try:
                selection_payload = json.loads(raw_selection)
            except (TypeError, json.JSONDecodeError):
                selection_payload = raw_selection
            if isinstance(selection_payload, list):
                raw_selection = json.dumps({"tool_calls": selection_payload})
            elif isinstance(selection_payload, dict):
                raw_selection = json.dumps(selection_payload)
            selection = parser.parse(
                raw_selection
            )
            available_tool_names = {tool.name for tool in self.tools}
            tool_calls = [
                {
                    "name": call.name,
                    "args": call.arguments,
                    "id": f"call_{uuid4().hex}",
                    "type": "tool_call",
                }
                for call in selection.tool_calls
                if call.name in available_tool_names
            ]
        except Exception as exc:
            log(f"graph.execute_task.tool_selection_fallback | error={exc!r}")
            tool_calls = []
        log(f"graph.execute_task.tool_calls | count={len(tool_calls)}")
        for call in tool_calls:
            log(
                f"graph.execute_task.tool_selected | name={call.get('name')} | args={call.get('args')}"
            )

        tools_by_name = {tool.name: tool for tool in self.tools}
        tool_records = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "unknown_tool")
            tool = tools_by_name.get(tool_name)
            try:
                if tool is None:
                    raise ValueError(f"Unknown research tool: {tool_name}")
                arguments = normalize_tool_arguments(tool_name, tool_call.get("args", {}))
                content = tool.invoke(arguments)
                source_hint = (
                    arguments.get("source_name")
                    if tool_name == "read_stored_file"
                    else tool_name
                )
                tool_records.extend(tool_content_to_records(content, source_hint))
            except Exception as exc:
                log(
                    f"graph.execute_task.tool_failed | name={tool_name}"
                    f" | error={exc!r}"
                )
                continue
        return {
            "tool_messages": state.get("tool_messages", [])
            + tool_records,
            "current_task_index": state.get("current_task_index", 0) + 1,
        }

    @log_function
    def collect_informations(self, state):
        """Aggregate tool responses, draft the answer, and store the selected sources."""
        sources = []
        for response in state.get("tool_messages", []):
            sources.append(response)
        rag_used = any(
            source.get("source", "").lower().endswith((".pdf", ".docx", ".txt", ".md"))
            or ", page " in source.get("source", "").lower()
            for source in sources
        )
        log(
            f"graph.sources.collected | rag_used={rag_used} | source_count={len(sources)}"
        )
        draft, recent_citations = create_draft_and_select_citations(
            self.llm, state, sources, self.session_memory.context()
        )
        citations = merge_citations(
            self.llm, state.get("citations", []), recent_citations, draft
        )
        return {
            "citations": citations,
            "tool_messages": [],
            "tasks": [],
            "draft_response": draft,
            "current_task_index": 0,
        }

    @log_function
    def clean_state(self, state):
        """Load the session conversation and clear transient graph data before a new run."""
        query = state.get("query", "")

        return {
            "query": query,
            "message_summary": state.get("message_summary", ""),
            "scope_category": "",
            "tasks": [],
            "current_task_index": 0,
            "tool_messages": [],
            "citations": [],
            "draft_response": "",
            "evaluation": {},
            "iterations": 0,
            "final_response": "",
            "hitl_answer": "",
            "needs_hitl": False,
        }

    @log_function
    def finalize_response(self, state):
        """Persist the completed turn and let the LLM update session memory outside graph state."""
        final_response = state.get("final_response") or state.get(
            "draft_response", "No answer was produced."
        )
        citations = state.get("citations", [])[:5]
        missing_citations = [citation for citation in citations if citation not in final_response]
        if missing_citations:
            final_response = (
                f"{final_response.rstrip()}\n\nSources:\n"
                + "\n".join(f"- {citation}" for citation in missing_citations)
            )
        self.session_memory.update_from_query(state.get("query", ""), self.llm)
        self.session_memory.update_from_response(final_response, self.llm)

        updated_state = {
            **state,
            "final_response": final_response,
            "messages": list(state.get("messages", [])),
            "tasks": [],
            "current_task_index": 0,
            "tool_messages": [],
            "citations": citations,
            "draft_response": "",
            "evaluation": state.get("evaluation", {}),
        }
        updated_state = append_recent_state_message(
            updated_state,
            {"role": "user", "content": state.get("query", "")},
            self.llm,
        )
        updated_state = append_recent_state_message(
            updated_state,
            {"role": "assistant", "content": final_response},
            self.llm,
        )
        return updated_state

    @log_function
    def evaluate_response(self, state):
        """Evaluate the draft and report whether further research is needed."""
        parser = llm_response_fixing_parser(ResponseEvaluation, self.llm)
        memory = self.session_memory.context()
        message_summary = state.get("message_summary", "")
        input_message = (
            f"Question:\n{state['query']}\n\n"
            f"Answer:\n{state.get('draft_response', '')}\n\n"
            f"User info: {memory.get('user_info', [])}\n"
            f"Session context: {memory.get('session_context', [])}\n"
            f"Older summarized context: {message_summary}\n"
            f"Recent conversation: {format_recent_messages(state.get('messages', []))}"
        )
        try:
            review = parser.parse(
                invoke_llm(
                    self.llm,
                    "evaluation_llm",
                    [
                        SystemMessage(content=EVALUATE_RESPONSE_SYSTEM_PROMPT),
                        HumanMessage(
                            content=f"{input_message}\n\n{parser.get_format_instructions()}"
                        ),
                    ],
                ).content
            )
            log(
                f"graph.evaluate_response.completed | improvement_scope_count={len(review.improvement_scopes)}"
            )
            return {
                "evaluation": review.model_dump(),
                "iterations": state.get("iterations", 0) + 1,
                "tasks": [],
                "current_task_index": 0,
                "tool_messages": [],
            }
        except Exception:
            return {
                "evaluation": {
                    "verdict": "Evaluation was unavailable.",
                    "needs_improvement": False,
                    "improvement_scopes": [],
                },
                "iterations": state.get("iterations", 0) + 1,
                "tasks": [],
                "current_task_index": 0,
                "tool_messages": [],
            }
