import json
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from .llm import build_llm
from .nodes import ResearchNodes
from .state import ResearchState
from .tools import build_research_tools
from .rag_system import DocumentHandler
from .config import settings
from .utils import log


class ResearchGraph(ResearchNodes):
    def __init__(self, rag, document_handler: DocumentHandler | None = None, llm=None, progress_callback=None):
        self.llm = llm or build_llm()
        self.progress_callback = progress_callback
        self.tools = build_research_tools(rag, document_handler=document_handler)
        self.tool_node = ToolNode(self.tools)
        self.graph = self._build().compile(checkpointer=MemorySaver())

    def _build(self):
        workflow = StateGraph(ResearchState)
        workflow.add_node("scope_gate", self.scope_gate)
        workflow.add_node("scope_response", self.scope_response)
        workflow.add_node("direct_answer", self.direct_answer)
        workflow.add_node("clarify", self.clarify)
        workflow.add_node("plan", self.plan)
        workflow.add_node("research_agent", self.research_agent)
        workflow.add_node("execute_tools", self.tool_node)
        workflow.add_node("collect_sources", self.collect_sources)
        workflow.add_node("clear_research_sources", self.clear_research_sources)
        workflow.add_node("draft", self.draft)
        workflow.add_node("critique", self.critique)
        workflow.add_node("revise", self.revise)
        workflow.add_edge(START, "scope_gate")
        workflow.add_conditional_edges(
            "scope_gate",
            self.route_scope,
            {"scope_response": "scope_response", "direct_answer": "direct_answer", "clarify": "clarify", "plan": "plan", "end": END},
        )
        workflow.add_edge("scope_response", "clear_research_sources")
        workflow.add_edge("direct_answer", "clear_research_sources")
        workflow.add_edge("clear_research_sources", END)
        workflow.add_conditional_edges(
            "clarify", self.route_clarification, {"plan": "plan", "end": "clear_research_sources"}
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
            "critique", self.route_quality, {"revise": "revise", "end": "clear_research_sources"}
        )
        workflow.add_edge("revise", "critique")
        return workflow

    def route_scope(self, state: ResearchState):
        if state.get("scope_category") == "out_of_scope":
            return "scope_response"
        if state.get("scope_category") == "answerable":
            return "direct_answer"
        if len(state.get("query", "").split()) < 5 and not state.get("hitl_answer"):
            return "clarify"
        return "plan"

    def _report(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(message)

    def route_clarification(self, state: ResearchState):
        return "plan" if state.get("hitl_answer") else "end"

    def route_tools(self, state: ResearchState):
        if state.get("tool_rounds", 0) >= 3:
            log("graph.routing.max_tool_rounds | next_step=collect_sources")
            return "collect_sources"
        messages = state.get("messages", [])
        tool_calls = messages[-1].tool_calls if messages else []
        if tool_calls:
            for call in tool_calls:
                tool_name = call.get("name", "research tool")
                query = call.get("args", {}).get("query", "")
                if query:
                    self._report(f"Searching {tool_name}: {query}")
                else:
                    self._report(f"Using {tool_name}")
            log(f"graph.routing.tools | next_step=execute_tools | call_count={len(tool_calls)}")
            return "execute_tools"
        log("graph.routing.no_tools | next_step=collect_sources | call_count=0")
        return "collect_sources"

    def route_after_tools(self, state: ResearchState):
        route = "research_agent" if state.get("tool_rounds", 0) < 3 else "collect_sources"
        log(f"graph.routing.after_tools | next_step={route} | round_count={state.get('tool_rounds', 0)}")
        return route

    def route_quality(self, state: ResearchState):
        if state.get("iterations", 0) >= getattr(settings, "max_research_iterations", 2):
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
            result["final_answer"] = result.get("final_answer", "")
        else:
            result["final_answer"] = result.get("draft", "No answer was produced.")
        return result
