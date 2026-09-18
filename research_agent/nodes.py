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
    RESEARCH_NODE_INPUT_TEMPLATE,
    RESEARCH_NODE_SYSTEM_PROMPT,
    SCOPE_GATE_SYSTEM_PROMPT,
    SINGLE_QUERY_INPUT_TEMPLATE,
    UNCLEAR_QUERY_INPUT_TEMPLATE,
    UNCLEAR_QUERY_RESPONSE_SYSTEM_PROMPT,
    AVAILABLE_TOOLS_TEMPLATE,
)
from .node_helpers import (
    create_draft_and_select_citations,
    format_recent_messages,
    invoke_llm,
    merge_citations,
    tool_messages_to_records,
)
from .utils import log, log_function


class ResearchNodes:
    @log_function
    def scope_gate(self, state):
        """Classify the request and route unrelated requests away from research."""
        parser = llm_response_fixing_parser(ScopeDecision, self.llm)
        input_message = f"User request:\n{state.get('query', '')}"
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
        input_message = OUT_OF_SCOPE_INPUT_TEMPLATE.format(
            query=state.get("query", ""),
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
        conversation = format_recent_messages(
            self.session_memory.context().get("recent_messages", [])
        )
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

        response = invoke_llm(
            self.llm,
            "unclear_query_response_llm",
            [
                SystemMessage(content=UNCLEAR_QUERY_RESPONSE_SYSTEM_PROMPT),
                HumanMessage(content=UNCLEAR_QUERY_INPUT_TEMPLATE.format(query=query)),
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
            log(f"graph.plan.created | task_count={len(plan.tasks)}")
            for index, task in enumerate(plan.tasks, start=1):
                log(f"graph.plan.task | index={index} | task={task!r}")
            return {"tasks": plan.tasks, "current_task_index": 0, "tool_responses": []}
        except Exception as exc:
            log(f"graph.plan.fallback | action=original_query | error={exc!r}")
            return {
                "tasks": [state["query"]],
                "current_task_index": 0,
                "tool_responses": [],
            }

    @log_function
    def research_node(self, state):
        """Ask the LLM for structured tool calls without requiring provider tool support."""
        tasks = state.get("tasks", [state["query"]])
        task_index = state.get("current_task_index", 0)
        task = tasks[task_index] if task_index < len(tasks) else state["query"]
        memory = self.session_memory.context()
        message_summary = state.get("message_summary", "")
        input_message = RESEARCH_NODE_INPUT_TEMPLATE.format(
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
            SystemMessage(content=RESEARCH_NODE_SYSTEM_PROMPT),
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
            selection = parser.parse(
                invoke_llm(self.llm, "research_node_llm", messages).content
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
            log(f"graph.research_node.tool_selection_fallback | error={exc!r}")
            tool_calls = []
        response = AIMessage(content="", tool_calls=tool_calls)
        log(f"graph.research_node.tool_calls | count={len(tool_calls)}")
        for call in tool_calls:
            log(
                f"graph.research_node.tool_selected | name={call.get('name')} | args={call.get('args')}"
            )
        return {"messages": [response], "tool_responses": []}

    @log_function
    def execute_tools(self, state):
        """Execute the tool calls selected by the research node and preserve their responses."""
        messages = state.get("messages", [])
        tool_calls = messages[-1].tool_calls if messages else []
        if not tool_calls:
            return {
                "tool_responses": [],
                "current_task_index": state.get("current_task_index", 0) + 1,
            }
        result = self.tool_node.invoke(state)
        tool_messages = result.get("messages", [])
        return {
            "messages": tool_messages,
            "tool_responses": state.get("tool_responses", [])
            + tool_messages_to_records(tool_messages),
            "current_task_index": state.get("current_task_index", 0) + 1,
        }

    @log_function
    def collect_informations(self, state):
        """Aggregate tool responses, draft the answer, and store the selected sources."""
        sources = []
        for response in state.get("tool_responses", []):
            sources.append(response)
        rag_used = any(source.get("source") == "rag_search" for source in sources)
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
            "tool_responses": [],
            "draft_response": draft,
            "current_task_index": 0,
        }

    def _append_recent_state_message(
        self, state: dict, message: dict[str, str]
    ) -> dict:
        """Keep at most 5 recent turns in graph state and fold removed turns into message_summary."""
        current_messages = list(state.get("messages", []))
        current_messages.append(message)

        if len(current_messages) <= 5:
            return {
                **state,
                "messages": current_messages,
                "message_summary": state.get("message_summary", ""),
            }

        removed = current_messages[:-5]
        remaining = current_messages[-5:]
        summary = state.get("message_summary", "").strip()
        removal_text = "\n".join(
            f"{entry.get('role', 'message')}: {entry.get('content', '')}"
            for entry in removed
            if entry.get("content")
        )
        if removal_text:
            summary = "\n".join(
                part for part in [summary, removal_text] if part
            ).strip()

        return {
            **state,
            "messages": remaining,
            "message_summary": summary,
        }

    @log_function
    def clean_state(self, state):
        """Load the session conversation and clear transient graph data before a new run."""
        query = state.get("query", "")
        state_messages = list(state.get("messages", []))
        if query and state_messages and state_messages[-1].get("content") != query:
            state_messages = self._append_recent_state_message(
                {
                    "messages": state_messages,
                    "message_summary": state.get("message_summary", ""),
                },
                {"role": "user", "content": query},
            )["messages"]
        elif query and not state_messages:
            state_messages = [{"role": "user", "content": query}]

        return {
            "query": query,
            "messages": state_messages,
            "message_summary": state.get("message_summary", ""),
            "scope_category": "",
            "tasks": [],
            "current_task_index": 0,
            "tool_responses": [],
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
        self.session_memory.update_from_query(state.get("query", ""), self.llm)
        self.session_memory.update_from_response(final_response, self.llm)

        updated_state = {
            **state,
            "final_response": final_response,
            "messages": list(state.get("messages", [])),
            "tasks": [],
            "current_task_index": 0,
            "tool_responses": [],
            "citations": state.get("citations", []),
            "draft_response": "",
            "evaluation": state.get("evaluation", {}),
        }
        updated_state = self._append_recent_state_message(
            updated_state,
            {"role": "user", "content": state.get("query", "")},
        )
        updated_state = self._append_recent_state_message(
            updated_state,
            {"role": "assistant", "content": final_response},
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
            }
        except Exception:
            return {
                "evaluation": {
                    "verdict": "Evaluation was unavailable.",
                    "improvement_scopes": [],
                },
                "iterations": state.get("iterations", 0) + 1,
            }
