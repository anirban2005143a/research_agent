import json
import re

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.types import interrupt

from .parsers import QualityReview, ResearchPlan, ScopeDecision, fixing_parser
from .prompts.rag import RAG_CONTEXT_LABEL
from .prompts.research import (
    CRITIQUE_SYSTEM,
    DRAFT_SYSTEM,
    DIRECT_ANSWER_SYSTEM,
    HITL_QUESTION,
    PLANNER_SYSTEM,
    RESEARCH_AGENT_SYSTEM,
    REVISION_SYSTEM,
    SCOPE_SYSTEM,
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
                "scope_reason": decision.reason,
                "scope_response": decision.response,
            }
        except Exception as exc:
            log(f"graph.scope_gate.fallback | action=research | error={exc!r}")
            return {"scope_allowed": True, "scope_category": "needs_research", "scope_reason": "Scope classification failed; research is safer."}

    @log_function
    def scope_response(self, state):
        return {"final_answer": state.get("scope_response", "Please provide a research question.")}

    @log_function
    def direct_answer(self, state):
        prompt = f"{DIRECT_ANSWER_SYSTEM}\nUser request: {state.get('query', '')}"
        response = self._invoke_llm("direct_answer_llm", [HumanMessage(content=prompt)])
        return {"final_answer": str(response.content).strip()}

    @log_function
    def clarify(self, state):
        answer = interrupt({"kind": "research_scope", "question": HITL_QUESTION})
        return {"hitl_answer": str(answer), "needs_hitl": False}

    @log_function
    def plan(self, state):
        parser = fixing_parser(ResearchPlan, self.llm)
        memory = state.get("memory_context", {})
        prompt = (
            f"{PLANNER_SYSTEM}\n{parser.get_format_instructions()}\n"
            f"Request: {state['query']}\nClarification: {state.get('hitl_answer', 'none')}\n"
            f"User preferences: {memory.get('preferences', {})}\n"
            f"Prior context summary: {memory.get('summary', '')}"
        )
        try:
            result = self._invoke_llm("planner_llm", [HumanMessage(content=prompt)])
            plan = parser.parse(result.content)
            log(f"graph.plan.created | step_count={len(plan.steps)}")
            for index, step in enumerate(plan.steps, start=1):
                log(f"graph.plan.step | index={index} | query={step!r}")
            return {"plan": plan.steps, "current_step": 0, "tool_rounds": 0}
        except Exception as exc:
            log(f"graph.plan.fallback | action=original_query | error={exc!r}")
            return {"plan": [state["query"]], "current_step": 0, "tool_rounds": 0}

    @log_function
    def research_agent(self, state):
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
        response = self._invoke_llm("research_agent_llm", messages, model=model)
        tool_calls = getattr(response, "tool_calls", [])
        log(f"graph.research_agent.tool_calls | count={len(tool_calls)}")
        for call in tool_calls:
            log(f"graph.research_agent.tool_selected | name={call.get('name')} | args={call.get('args')}")
        return {"messages": [response], "tool_rounds": state.get("tool_rounds", 0) + 1}

    @log_function
    def collect_sources(self, state):
        sources = []
        for message in state.get("messages", []):
            if isinstance(message, ToolMessage):
                content = str(message.content)
                try:
                    payload = json.loads(content)
                except json.JSONDecodeError:
                    payload = None

                if isinstance(payload, dict) and payload.get("result_type") == "rag_search_results":
                    for result in payload.get("results", []):
                        sources.append(
                            {
                                "source": result.get("source", message.name or "rag_search"),
                                "content": result.get("content", ""),
                                "locations": [result.get("citation", "")],
                                "tool_call_id": message.tool_call_id,
                            }
                        )
                    continue

                urls = re.findall(r"https?://[^\s)]+", content)
                citations = re.findall(r"Citation:\s*([^\n]+)", content)
                locations = citations or urls
                sources.append(
                    {
                        "source": message.name or "research tool",
                        "content": content,
                        "locations": locations,
                        "tool_call_id": message.tool_call_id,
                    }
                )
        context = list(sources)
        rag_used = any(source.get("source") == "rag_search" for source in sources)
        log(f"graph.sources.collected | rag_used={rag_used} | external_count={len(sources)} | evidence_count={len(context)}")
        return {"sources": sources, "retrieved_context": context}

    @log_function
    def clear_research_sources(self, state):
        """Clear per-run citation context after the final answer is produced."""
        return {"sources": [], "retrieved_context": []}

    @log_function
    def draft(self, state):
        evidence_blocks = []
        for index, item in enumerate(state.get("retrieved_context", []), start=1):
            locations = item.get("locations", [])
            location = "; ".join(locations) or item.get("source", "")
            evidence_blocks.append(
                f"[{index}] Source: {item.get('source', '')}\n"
                f"Location: {location}\n"
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
        answer = self._invoke_llm("draft_llm", [prompt, HumanMessage(content=user_prompt)]).content
        log(f"graph.draft.completed | response_chars={len(str(answer))}")
        return {"draft": str(answer), "iterations": state.get("iterations", 0)}

    @log_function
    def critique(self, state):
        parser = fixing_parser(QualityReview, self.llm)
        prompt = f"{CRITIQUE_SYSTEM}\n{parser.get_format_instructions()}\nQuestion: {state['query']}\nAnswer:\n{state.get('draft', '')}"
        try:
            review = parser.parse(self._invoke_llm("critique_llm", [HumanMessage(content=prompt)]).content)
            log(f"graph.critique.completed | score={review.score} | needs_more_research={review.needs_more_research} | issue_count={len(review.issues)}")
            return {
                "critique": review.model_dump_json(),
                "iterations": state.get("iterations", 0) + 1,
            }
        except Exception:
            return {
                "critique": json.dumps({"needs_more_research": False, "issues": ["Review unavailable"]}),
                "iterations": state.get("iterations", 0) + 1,
            }

    @log_function
    def revise(self, state):
        prompt = HumanMessage(
            content=f"{REVISION_SYSTEM}\nReview: {state.get('critique', '')}\nDraft: {state.get('draft', '')}"
        )
        answer = self._invoke_llm("revision_llm", [prompt]).content
        log(f"graph.revision.completed | response_chars={len(str(answer))}")
        return {"draft": str(answer)}

    def _invoke_llm(self, label: str, messages, model=None):
        response = retry_call(lambda: (model or self.llm).invoke(messages), label)
        log(f"llm.completed | name={label} | response_chars={len(str(getattr(response, 'content', response)))}")
        return response
