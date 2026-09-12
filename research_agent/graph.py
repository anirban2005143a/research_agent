import json
import time
from functools import wraps
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from .llm import build_llm
from .parsers import QualityReview, ResearchPlan, fixing_parser
from .prompts.rag import RAG_CONTEXT_LABEL
from .prompts.research import (
    CRITIQUE_SYSTEM,
    DRAFT_SYSTEM,
    HITL_QUESTION,
    PLANNER_SYSTEM,
    RESEARCH_AGENT_SYSTEM,
    REVISION_SYSTEM,
)
from .scope import RESEARCH_ONLY_REPLY, classify_query
from .state import ResearchState
from .tools import build_research_tools
from .config import settings
from .observability import log


def log_node(function):
    @wraps(function)
    def wrapped(self, state):
        started = time.perf_counter()
        self._report(f"{function.__name__.replace('_', ' ').title()} in progress")
        log(f"NODE {function.__name__} | status=started")
        try:
            result = function(self, state)
            log(f"NODE {function.__name__} | status=completed | elapsed={time.perf_counter() - started:.2f}s")
            return result
        except Exception as exc:
            log(f"NODE {function.__name__} | status=failed | elapsed={time.perf_counter() - started:.2f}s | error={exc!r}")
            raise
    return wrapped


class ResearchGraph:
    def __init__(self, rag, llm=None, progress_callback=None):
        self.rag = rag
        self.llm = llm or build_llm()
        self.progress_callback = progress_callback
        self.tools = build_research_tools(rag)
        self.tool_node = ToolNode(self.tools)
        self.graph = self._build().compile(checkpointer=MemorySaver())

    def _build(self):
        workflow = StateGraph(ResearchState)
        workflow.add_node("scope_gate", self.scope_gate)
        workflow.add_node("clarify", self.clarify)
        workflow.add_node("plan", self.plan)
        workflow.add_node("research_agent", self.research_agent)
        workflow.add_node("execute_tools", self.tool_node)
        workflow.add_node("collect_sources", self.collect_sources)
        workflow.add_node("draft", self.draft)
        workflow.add_node("critique", self.critique)
        workflow.add_node("revise", self.revise)
        workflow.add_edge(START, "scope_gate")
        workflow.add_conditional_edges(
            "scope_gate",
            self.route_scope,
            {"clarify": "clarify", "plan": "plan", "end": END},
        )
        workflow.add_conditional_edges(
            "clarify", self.route_clarification, {"plan": "plan", "end": END}
        )
        workflow.add_edge("plan", "research_agent")
        workflow.add_conditional_edges(
            "research_agent",
            self.route_tools,
            {"execute_tools": "execute_tools", "collect_sources": "collect_sources"},
        )
        workflow.add_conditional_edges(
            "execute_tools",
            self.route_after_tools,
            {"research_agent": "research_agent", "collect_sources": "collect_sources"},
        )
        workflow.add_edge("collect_sources", "draft")
        workflow.add_edge("draft", "critique")
        workflow.add_conditional_edges(
            "critique", self.route_quality, {"revise": "revise", "end": END}
        )
        workflow.add_edge("revise", "critique")
        return workflow

    @log_node
    def scope_gate(self, state: ResearchState):
        allowed, reason = classify_query(state.get("query", ""))
        return {"scope_allowed": allowed, "scope_reason": reason}

    def _report(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(message)

    def route_scope(self, state: ResearchState):
        if not state.get("scope_allowed"):
            return "end"
        if len(state.get("query", "").split()) < 5 and not state.get("hitl_answer"):
            return "clarify"
        return "plan"

    @log_node
    def clarify(self, state: ResearchState):
        answer = interrupt({"kind": "research_scope", "question": HITL_QUESTION})
        return {"hitl_answer": str(answer), "needs_hitl": False}

    def route_clarification(self, state: ResearchState):
        return "plan" if state.get("hitl_answer") else "end"

    @log_node
    def plan(self, state: ResearchState):
        parser = fixing_parser(ResearchPlan, self.llm)
        memory = state.get("memory_context", {})
        prompt = (
            f"{PLANNER_SYSTEM}\n{parser.get_format_instructions()}\n"
            f"Request: {state['query']}\nClarification: {state.get('hitl_answer', 'none')}\n"
            f"User preferences: {memory.get('preferences', {})}\n"
            f"Prior context summary: {memory.get('summary', '')}"
        )
        try:
            result = self._invoke_llm(
                "planner_llm", [HumanMessage(content=prompt)]
            )
            plan = parser.parse(result.content)
            log(f"PLANNER | count={len(plan.steps)}")
            for index, step in enumerate(plan.steps, start=1):
                log(f"PLANNER | step={index} | executing_query={step!r}")
            return {"plan": plan.steps, "current_step": 0, "tool_rounds": 0}
        except Exception as exc:
            log(f"PLANNER | fallback_to_original_query | error={exc!r}")
            return {"plan": [state["query"]], "current_step": 0, "tool_rounds": 0}

    @log_node
    def research_agent(self, state: ResearchState):
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
        log(f"RESEARCH_AGENT | tool_call_count={len(tool_calls)}")
        for call in tool_calls:
            log(f"RESEARCH_AGENT | selected_tool={call.get('name')} | args={call.get('args')}")
        return {"messages": [response], "tool_rounds": state.get("tool_rounds", 0) + 1}

    def route_tools(self, state: ResearchState):
        if state.get("tool_rounds", 0) >= 3:
            log("ROUTER | maximum tool rounds reached | route=collect_sources")
            return "collect_sources"
        messages = state.get("messages", [])
        tool_calls = messages[-1].tool_calls if messages else []
        if tool_calls:
            self._report("Selecting and executing research sources")
            log(f"ROUTER | route=execute_tools | calls={len(tool_calls)}")
            return "execute_tools"
        log("ROUTER | route=collect_sources | calls=0")
        return "collect_sources"

    def route_after_tools(self, state: ResearchState):
        route = "research_agent" if state.get("tool_rounds", 0) < 3 else "collect_sources"
        log(f"ROUTER | after_tools={route} | tool_rounds={state.get('tool_rounds', 0)}")
        return route

    @log_node
    def collect_sources(self, state: ResearchState):
        sources = []
        for message in state.get("messages", []):
            if isinstance(message, ToolMessage):
                sources.append(
                    {
                        "source": message.name or "research tool",
                        "content": str(message.content),
                        "tool_call_id": message.tool_call_id,
                    }
                )
        context = list(sources)
        if self.rag:
            context.extend(self.rag.retrieve(state["query"]))
        log(f"SOURCES | external={len(sources)} | total_evidence={len(context)}")
        return {"sources": sources, "retrieved_context": context}

    @log_node
    def draft(self, state: ResearchState):
        evidence_blocks = []
        for index, item in enumerate(state.get("retrieved_context", []), start=1):
            evidence_blocks.append(
                f"[{index}] Source: {item.get('source', '')}\n"
                f"Location: {item.get('citation', item.get('source', ''))}\n"
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
        answer = self._invoke_llm(
            "draft_llm", [prompt, HumanMessage(content=user_prompt)]
        ).content
        log(f"DRAFT | response_chars={len(str(answer))}")
        return {"draft": str(answer), "iterations": state.get("iterations", 0)}

    @log_node
    def critique(self, state: ResearchState):
        parser = fixing_parser(QualityReview, self.llm)
        prompt = f"{CRITIQUE_SYSTEM}\n{parser.get_format_instructions()}\nQuestion: {state['query']}\nAnswer:\n{state.get('draft', '')}"
        try:
            review = parser.parse(self._invoke_llm("critique_llm", [HumanMessage(content=prompt)]).content)
            log(f"CRITIQUE | score={review.score} | needs_more_research={review.needs_more_research} | issues={len(review.issues)}")
            return {
                "critique": review.model_dump_json(),
                "iterations": state.get("iterations", 0) + 1,
            }
        except Exception:
            return {
                "critique": json.dumps(
                    {"needs_more_research": False, "issues": ["Review unavailable"]}
                ),
                "iterations": state.get("iterations", 0) + 1,
            }

    def route_quality(self, state: ResearchState):
        if state.get("iterations", 0) >= 2:
            return "end"
        try:
            return (
                "revise"
                if json.loads(state.get("critique", "{}")).get(
                    "needs_more_research", False
                )
                else "end"
            )
        except json.JSONDecodeError:
            return "end"

    @log_node
    def revise(self, state: ResearchState):
        prompt = HumanMessage(
            content=f"{REVISION_SYSTEM}\nReview: {state.get('critique', '')}\nDraft: {state.get('draft', '')}"
        )
        answer = self._invoke_llm("revision_llm", [prompt]).content
        log(f"REVISION | response_chars={len(str(answer))}")
        return {"draft": str(answer)}

    def _invoke_llm(self, label: str, messages, model=None):
        log(f"LLM {label} | throttle_sleep={settings.llm_call_delay_seconds}s")
        time.sleep(settings.llm_call_delay_seconds)
        started = time.perf_counter()
        response = (model or self.llm).invoke(messages)
        log(f"LLM {label} | response_time={time.perf_counter() - started:.2f}s | response_chars={len(str(getattr(response, 'content', response)))}")
        return response

    def invoke(
        self,
        query: str,
        messages: list[Any] | None = None,
        hitl_answer: str = "",
        thread_id: str = "default",
        memory_context: dict[str, Any] | None = None,
        progress_callback=None,
    ):
        if progress_callback:
            self.progress_callback = progress_callback
        config = {"configurable": {"thread_id": thread_id}}
        if hitl_answer:
            result = self.graph.invoke(Command(resume=hitl_answer), config=config)
        else:
            result = self.graph.invoke(
                {
                    "session_id": thread_id,
                    "query": query,
                    "messages": messages or [],
                    "memory_context": memory_context or {},
                },
                config=config,
            )
        if result.get("__interrupt__"):
            result["needs_hitl"] = True
            result["hitl_question"] = result["__interrupt__"][0].value["question"]
            result["final_answer"] = result["hitl_question"]
        elif not result.get("scope_allowed"):
            result["final_answer"] = RESEARCH_ONLY_REPLY
        else:
            result["final_answer"] = result.get("draft", "No answer was produced.")
        return result
