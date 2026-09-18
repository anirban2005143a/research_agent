import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from .parsers import (
    ClarificationDecision,
    ResearchDraft,
    ResearchPlan,
    ResponseEvaluation,
    ScopeDecision,
    fixing_parser,
)
from .prompts.rag import RAG_CONTEXT_LABEL
from .prompts.research import (
    DRAFT_SYSTEM,
    EVALUATION_SYSTEM,
    HITL_QUESTION,
    CLARIFICATION_SYSTEM,
    OUT_OF_SCOPE_RESPONSE_SYSTEM,
    PLANNER_SYSTEM,
    RESEARCH_AGENT_SYSTEM,
    SCOPE_SYSTEM,
    UNCLEAR_QUERY_RESPONSE_SYSTEM,
)
from .utils import log, log_function, retry_call


class ResearchNodes:
    @log_function
    def scope_gate(self, state):
        parser = fixing_parser(ScopeDecision, self.llm)
        prompt = f"{SCOPE_SYSTEM}\n{parser.get_format_instructions()}\nUser request: {state.get('query', '')}"
        try:
            decision = parser.parse(self._invoke_llm("scope_gate_llm", [HumanMessage(content=prompt)]).content)
            category = decision.category if decision.category in {"out_of_scope", "answerable", "needs_research"} else "needs_research"
            return {
                "scope_allowed": category != "out_of_scope",
                "scope_category": category,
                "out_of_scope_response": decision.response,
            }
        except Exception as exc:
            log(f"graph.scope_gate.fallback | action=research | error={exc!r}")
            return {"scope_allowed": True, "scope_category": "needs_research"}

    @log_function
    def out_of_scope_response(self, state):
        prompt = (
            f"{OUT_OF_SCOPE_RESPONSE_SYSTEM}\n"
            f"User request: {state.get('query', '')}\n"
            f"Scope assessment: {state.get('out_of_scope_response', '')}"
        )
        response = self._invoke_llm("out_of_scope_response_llm", [HumanMessage(content=prompt)])
        return {"final_answer": str(response.content).strip(), "citations": []}

    @log_function
    def clarify_query(self, state):
        parser = fixing_parser(ClarificationDecision, self.llm)
        query = state.get("query", "")
        hitl_answer = str(state.get("hitl_answer", "")).strip()
        if hitl_answer:
            prompt = (
                f"{CLARIFICATION_SYSTEM}\n{parser.get_format_instructions()}\n"
                f"Original query: {query}\nClarification answer: {hitl_answer}"
            )
            try:
                decision = parser.parse(
                    self._invoke_llm("clarified_query_llm", [HumanMessage(content=prompt)]).content
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

        prompt = (
            f"{CLARIFICATION_SYSTEM}\n{parser.get_format_instructions()}\n"
            f"Original query: {query}"
        )
        try:
            decision = parser.parse(
                self._invoke_llm("clarify_query_llm", [HumanMessage(content=prompt)]).content
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

        question = decision.question.strip() or HITL_QUESTION
        answer = interrupt({"kind": "research_scope", "question": question})
        answer_text = "" if answer is None else str(answer).strip()
        if answer_text:
            return {"hitl_answer": answer_text, "needs_hitl": False}

        response = self._invoke_llm(
            "unclear_query_response_llm",
            [HumanMessage(content=f"{UNCLEAR_QUERY_RESPONSE_SYSTEM}\nUser query: {query}")],
        )
        return {"final_answer": str(response.content).strip(), "citations": [], "needs_hitl": False}

    @log_function
    def plan(self, state):
        parser = fixing_parser(ResearchPlan, self.llm)
        memory = state.get("memory_context", {})
        prompt = (
            f"{PLANNER_SYSTEM}\n{parser.get_format_instructions()}\n"
            f"Request: {state['query']}\nClarification: {state.get('hitl_answer', 'none')}\n"
            f"Current draft: {state.get('draft', 'No draft exists yet.')}\n"
            f"Evaluation result: {state.get('evaluation', 'No evaluation exists yet.')}\n"
            f"User preferences: {memory.get('preferences', {})}\n"
            f"Prior context summary: {memory.get('summary', '')}"
        )
        try:
            result = self._invoke_llm("planner_llm", [HumanMessage(content=prompt)])
            plan = parser.parse(result.content)
            log(f"graph.plan.created | step_count={len(plan.steps)}")
            for index, step in enumerate(plan.steps, start=1):
                log(f"graph.plan.step | index={index} | query={step!r}")
            return {"plan": plan.steps, "tool_responses": []}
        except Exception as exc:
            log(f"graph.plan.fallback | action=original_query | error={exc!r}")
            return {"plan": [state["query"]], "tool_responses": []}

    @log_function
    def research_node(self, state):
        model = self.llm.bind_tools(self.tools)
        plan = "\n".join(f"- {step}" for step in state.get("plan", [state["query"]]))
        memory = state.get("memory_context", {})
        messages = [
            SystemMessage(content=RESEARCH_AGENT_SYSTEM),
            HumanMessage(
                content=(
                    f"Research request: {state['query']}\nResearch plan:\n{plan}\n"
                    f"User preferences: {memory.get('preferences', {})}\n"
                    f"Prior context summary: {memory.get('summary', '')}"
                )
            ),
        ]
        messages.extend(state.get("messages", []))
        response = self._invoke_llm("research_node_llm", messages, model=model)
        tool_calls = getattr(response, "tool_calls", [])
        log(f"graph.research_node.tool_calls | count={len(tool_calls)}")
        for call in tool_calls:
            log(f"graph.research_node.tool_selected | name={call.get('name')} | args={call.get('args')}")
        return {"messages": [response]}

    @log_function
    def execute_tools(self, state):
        messages = state.get("messages", [])
        tool_calls = messages[-1].tool_calls if messages else []
        for call in tool_calls:
            self._report(f"Using {call.get('name', 'research tool')}")
        if not tool_calls:
            return {"tool_responses": []}
        result = self.tool_node.invoke(state)
        tool_messages = result.get("messages", [])
        return {
            "messages": tool_messages,
            "tool_responses": self._tool_messages_to_records(tool_messages),
        }

    @log_function
    def collect_informations(self, state):
        sources = []
        for response in state.get("tool_responses", []):
            source_id = f"source-{len(sources) + 1}"
            sources.append({**response, "source_id": source_id})
        rag_used = any(source.get("source") == "rag_search" for source in sources)
        log(f"graph.sources.collected | rag_used={rag_used} | source_count={len(sources)}")
        return {"sources": sources, "citations": [], "tool_responses": []}

    @log_function
    def clear_citations(self, state):
        """Clear per-run citation context after the final answer is produced."""
        return {"sources": [], "citations": [], "tool_responses": []}

    @log_function
    def draft_response(self, state):
        evidence_blocks = []
        for item in state.get("sources", []):
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
        prompt = SystemMessage(content=DRAFT_SYSTEM)
        memory = state.get("memory_context", {})
        user_prompt = (
            f"{RAG_CONTEXT_LABEL}\n\nQuestion: {state['query']}\n"
            f"User preferences: {memory.get('preferences', {})}\n"
            f"Prior context summary: {memory.get('summary', '')}\n\n"
            f"Evidence:\n{context or 'No external evidence was found.'}"
        )
        parser = fixing_parser(ResearchDraft, self.llm)
        try:
            result = parser.parse(
                self._invoke_llm(
                    "draft_llm",
                    [prompt, HumanMessage(content=f"{user_prompt}\n\n{parser.get_format_instructions()}")],
                ).content
            )
            answer = result.answer
            citations = self._select_citations(result.citation_ids, state.get("sources", []))
        except Exception as exc:
            log(f"graph.draft_response.parse_fallback | error={exc!r}")
            answer = str(self._invoke_llm("draft_response_llm_fallback", [prompt, HumanMessage(content=user_prompt)]).content)
            citations = []
        log(f"graph.draft_response.completed | response_chars={len(str(answer))} | citation_count={len(citations)}")
        return {"draft": str(answer), "citations": citations, "iterations": state.get("iterations", 0)}

    @log_function
    def evaluate_response(self, state):
        parser = fixing_parser(ResponseEvaluation, self.llm)
        prompt = f"{EVALUATION_SYSTEM}\n{parser.get_format_instructions()}\nQuestion: {state['query']}\nAnswer:\n{state.get('draft', '')}"
        try:
            review = parser.parse(self._invoke_llm("evaluation_llm", [HumanMessage(content=prompt)]).content)
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

    @staticmethod
    def _select_citations(citation_ids, sources):
        selected_ids = set(citation_ids)
        return [source for source in sources if source.get("source_id") in selected_ids]

    @staticmethod
    def _tool_messages_to_records(tool_messages):
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
