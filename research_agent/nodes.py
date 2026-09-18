import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from .parsers import (
    ClarificationDecision,
    ResearchDraft,
    ResearchPlan,
    ResponseEvaluation,
    ScopeCategory,
    ScopeDecision,
    llm_response_fixing_parser,
)
from .prompts import (
    CLARIFY_QUERY_INPUT_TEMPLATE,
    CLARIFY_QUERY_SYSTEM_PROMPT,
    DRAFT_RESPONSE_INPUT_TEMPLATE,
    DRAFT_RESPONSE_SYSTEM_PROMPT,
    EVALUATE_RESPONSE_SYSTEM_PROMPT,
    HITL_CLARIFICATION_QUESTION,
    OUT_OF_SCOPE_INPUT_TEMPLATE,
    OUT_OF_SCOPE_RESPONSE_SYSTEM_PROMPT,
    PLANNING_INPUT_TEMPLATE,
    PLANNING_SYSTEM_PROMPT,
    RAG_EVIDENCE_CONTEXT,
    RESEARCH_NODE_INPUT_TEMPLATE,
    RESEARCH_NODE_SYSTEM_PROMPT,
    SCOPE_GATE_SYSTEM_PROMPT,
    SINGLE_QUERY_INPUT_TEMPLATE,
    UNCLEAR_QUERY_INPUT_TEMPLATE,
    UNCLEAR_QUERY_RESPONSE_SYSTEM_PROMPT,
)
from .utils import log, log_function, retry_call


class ResearchNodes:
    @log_function
    def scope_gate(self, state):
        """Classify the request and route unrelated requests away from research."""
        parser = llm_response_fixing_parser(ScopeDecision, self.llm)
        input_message = f"User request:\n{state.get('query', '')}"
        try:
            decision = parser.parse(
                self._invoke_llm(
                    "scope_gate_llm",
                    [
                        SystemMessage(content=SCOPE_GATE_SYSTEM_PROMPT),
                        HumanMessage(content=f"{input_message}\n\n{parser.get_format_instructions()}"),
                    ],
                ).content
            )
            category = decision.category
            return {
                "scope_allowed": category != ScopeCategory.OUT_OF_SCOPE,
                "scope_category": category.value,
            }
        except Exception as exc:
            log(f"graph.scope_gate.fallback | action=research | error={exc!r}")
            return {"scope_allowed": True, "scope_category": "needs_research"}

    @log_function
    def out_of_scope_response(self, state):
        """Explain the research scope and guide an unrelated request toward a research question."""
        input_message = OUT_OF_SCOPE_INPUT_TEMPLATE.format(
            query=state.get("query", ""),
        )
        response = self._invoke_llm(
            "out_of_scope_response_llm",
            [
                SystemMessage(content=OUT_OF_SCOPE_RESPONSE_SYSTEM_PROMPT),
                HumanMessage(content=input_message),
            ],
        )
        return {"final_answer": str(response.content).strip(), "citations": []}

    @log_function
    def clarify_query(self, state):
        """Decide whether clarification is needed and store a clean final query."""
        parser = llm_response_fixing_parser(ClarificationDecision, self.llm)
        query = state.get("query", "")
        hitl_answer = str(state.get("hitl_answer", "")).strip()
        if hitl_answer:
            input_message = CLARIFY_QUERY_INPUT_TEMPLATE.format(
                query=query, clarification_answer=hitl_answer
            )
            try:
                decision = parser.parse(
                    self._invoke_llm(
                        "clarified_query_llm",
                        [
                            SystemMessage(content=CLARIFY_QUERY_SYSTEM_PROMPT),
                            HumanMessage(content=f"{input_message}\n\n{parser.get_format_instructions()}"),
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

        input_message = SINGLE_QUERY_INPUT_TEMPLATE.format(query=query)
        try:
            decision = parser.parse(
                self._invoke_llm(
                    "clarify_query_llm",
                    [
                        SystemMessage(content=CLARIFY_QUERY_SYSTEM_PROMPT),
                        HumanMessage(content=f"{input_message}\n\n{parser.get_format_instructions()}"),
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

        response = self._invoke_llm(
            "unclear_query_response_llm",
            [
                SystemMessage(content=UNCLEAR_QUERY_RESPONSE_SYSTEM_PROMPT),
                HumanMessage(content=UNCLEAR_QUERY_INPUT_TEMPLATE.format(query=query)),
            ],
        )
        return {"final_answer": str(response.content).strip(), "citations": [], "needs_hitl": False}

    @log_function
    def plan(self, state):
        """Create the next evidence-gathering plan from the query and current evaluation."""
        parser = llm_response_fixing_parser(ResearchPlan, self.llm)
        memory = state.get("memory_context", {})
        input_message = PLANNING_INPUT_TEMPLATE.format(
            query=state["query"],
            clarification=state.get("hitl_answer", "none"),
            draft=state.get("draft", "No draft exists yet."),
            evaluation=state.get("evaluation", "No evaluation exists yet."),
            preferences=memory.get("preferences", {}),
            summary=memory.get("summary", ""),
        )
        try:
            result = self._invoke_llm(
                "planner_llm",
                [
                    SystemMessage(content=PLANNING_SYSTEM_PROMPT),
                    HumanMessage(content=f"{input_message}\n\n{parser.get_format_instructions()}"),
                ],
            )
            plan = parser.parse(result.content)
            log(f"graph.plan.created | task_count={len(plan.tasks)}")
            for index, task in enumerate(plan.tasks, start=1):
                log(f"graph.plan.task | index={index} | task={task!r}")
            return {"tasks": plan.tasks, "current_task_index": 0, "tool_responses": []}
        except Exception as exc:
            log(f"graph.plan.fallback | action=original_query | error={exc!r}")
            return {"tasks": [state["query"]], "current_task_index": 0, "tool_responses": []}

    @log_function
    def research_node(self, state):
        """Ask the LLM to select all tool calls required by the current research plan."""
        model = self.llm.bind_tools(self.tools)
        tasks = state.get("tasks", [state["query"]])
        task_index = state.get("current_task_index", 0)
        task = tasks[task_index] if task_index < len(tasks) else state["query"]
        memory = state.get("memory_context", {})
        input_message = RESEARCH_NODE_INPUT_TEMPLATE.format(
            query=state["query"],
            task=task,
            preferences=memory.get("preferences", {}),
            summary=memory.get("summary", ""),
        )
        messages = [
            SystemMessage(content=RESEARCH_NODE_SYSTEM_PROMPT),
            HumanMessage(content=input_message),
        ]
        messages.extend(state.get("messages", []))
        response = self._invoke_llm("research_node_llm", messages, model=model)
        tool_calls = getattr(response, "tool_calls", [])
        log(f"graph.research_node.tool_calls | count={len(tool_calls)}")
        for call in tool_calls:
            log(f"graph.research_node.tool_selected | name={call.get('name')} | args={call.get('args')}")
        return {"messages": [response], "tool_responses": []}

    @log_function
    def execute_tools(self, state):
        """Execute the tool calls selected by the research node and preserve their responses."""
        messages = state.get("messages", [])
        tool_calls = messages[-1].tool_calls if messages else []
        for call in tool_calls:
            self._report(f"Using {call.get('name', 'research tool')}")
        if not tool_calls:
            return {"tool_responses": [], "current_task_index": state.get("current_task_index", 0) + 1}
        result = self.tool_node.invoke(state)
        tool_messages = result.get("messages", [])
        return {
            "messages": tool_messages,
            "tool_responses": state.get("tool_responses", []) + self._tool_messages_to_records(tool_messages),
            "current_task_index": state.get("current_task_index", 0) + 1,
        }

    @log_function
    def collect_informations(self, state):
        """Aggregate tool responses, draft the answer, and store the selected sources."""
        sources = []
        for response in state.get("tool_responses", []):
            source_id = f"source-{len(sources) + 1}"
            sources.append({**response, "source_id": source_id})
        rag_used = any(source.get("source") == "rag_search" for source in sources)
        log(f"graph.sources.collected | rag_used={rag_used} | source_count={len(sources)}")
        draft, citations = self._create_draft_and_select_citations(state, sources)
        return {
            "sources": sources,
            "citations": citations,
            "tool_responses": [],
            "draft": draft,
            "current_task_index": 0,
        }

    @log_function
    def clear_citations(self, state):
        """Clear temporary evidence and citation records before graph termination."""
        return {"sources": [], "citations": [], "tool_responses": []}

    def _create_draft_and_select_citations(self, state, sources):
        """Create a draft from aggregated sources and return the LLM-selected source records."""
        evidence_blocks = []
        for item in sources:
            locations = item.get("locations", [])
            location = "; ".join(locations) or item.get("source", "")
            evidence_blocks.append(
                f"Source ID: {item.get('source_id', '')}\n"
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
        parser = llm_response_fixing_parser(ResearchDraft, self.llm)
        try:
            result = parser.parse(
                self._invoke_llm(
                    "draft_llm",
                    [
                        SystemMessage(content=DRAFT_RESPONSE_SYSTEM_PROMPT),
                        HumanMessage(content=f"{input_message}\n\n{parser.get_format_instructions()}"),
                    ],
                ).content
            )
            answer = result.answer
            citations = self._select_citations(result.citation_ids, state.get("sources", []))
        except Exception as exc:
            log(f"graph.collect_informations.draft_fallback | error={exc!r}")
            answer = str(
                self._invoke_llm(
                    "draft_response_llm_fallback",
                    [SystemMessage(content=DRAFT_RESPONSE_SYSTEM_PROMPT), HumanMessage(content=input_message)],
                ).content
            )
            citations = []
        log(f"graph.collect_informations.completed | response_chars={len(str(answer))} | citation_count={len(citations)}")
        return str(answer), citations

    @log_function
    def evaluate_response(self, state):
        """Evaluate the draft and report whether further research is needed."""
        parser = llm_response_fixing_parser(ResponseEvaluation, self.llm)
        input_message = f"Question:\n{state['query']}\n\nAnswer:\n{state.get('draft', '')}"
        try:
            review = parser.parse(
                self._invoke_llm(
                    "evaluation_llm",
                    [
                        SystemMessage(content=EVALUATE_RESPONSE_SYSTEM_PROMPT),
                        HumanMessage(content=f"{input_message}\n\n{parser.get_format_instructions()}"),
                    ],
                ).content
            )
            log(f"graph.evaluate_response.completed | score={review.score} | needs_more_research={review.needs_more_research} | issue_count={len(review.issues)}")
            return {
                "evaluation": review.model_dump_json(),
                "iterations": state.get("iterations", 0) + 1,
            }
        except Exception:
            return {
                "evaluation": json.dumps({"needs_more_research": False, "issues": ["Review unavailable"]}),
                "iterations": state.get("iterations", 0) + 1,
            }

    def _select_citations(self, citation_ids, sources):
        selected_ids = set(citation_ids)
        return [source for source in sources if source.get("source_id") in selected_ids]

    def _tool_messages_to_records(self, tool_messages):
        records = []
        for message in tool_messages:
            content = str(message.content)
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
                    "content": content,
                    "locations": re.findall(r"https?://[^\s)]+", content),
                    "metadata": {
                        "tool": message.name or "research tool",
                        "tool_call_id": message.tool_call_id,
                    },
                    "tool_call_id": message.tool_call_id,
                }
            )
        return records

    def _invoke_llm(self, label: str, messages, model=None):
        response = retry_call(lambda: (model or self.llm).invoke(messages), label)
        log(f"llm.completed | name={label} | response_chars={len(str(getattr(response, 'content', response)))}")
        return response
