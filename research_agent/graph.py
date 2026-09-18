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
from .utils import log


class ResearchGraph(ResearchNodes):
    def __init__(self, rag, document_handler: DocumentHandler | None = None, llm=None):
        self.llm = llm or build_llm()
        self.tools = build_research_tools(rag, document_handler=document_handler)
        self.tool_node = ToolNode(self.tools)
        self.graph = self._build().compile(checkpointer=MemorySaver())

    def _build(self):
        workflow = StateGraph(ResearchState)
        workflow.add_node("clean_state", self.clean_state)
        workflow.add_node("scope_gate", self.scope_gate)
        workflow.add_node("out_of_scope_response", self.out_of_scope_response)
        workflow.add_node("clarify_query", self.clarify_query)
        workflow.add_node("plan", self.plan)
        workflow.add_node("research_node", self.research_node)
        workflow.add_node("execute_tools", self.execute_tools)
        workflow.add_node("collect_informations", self.collect_informations)
        workflow.add_node("evaluate_response", self.evaluate_response)
        workflow.add_node("finalize_response", self.finalize_response)
        workflow.add_edge(START, "clean_state")
        workflow.add_edge("clean_state", "scope_gate")
        workflow.add_conditional_edges(
            "scope_gate",
            self.route_scope,
            {"out_of_scope_response": "out_of_scope_response", "clarify_query": "clarify_query"},
        )
        workflow.add_edge("out_of_scope_response", "finalize_response")
        workflow.add_conditional_edges(
            "clarify_query",
            self.route_clarification,
            {"clarify_query": "clarify_query", "plan": "plan", "finalize_response": "finalize_response"},
        )
        workflow.add_edge("plan", "research_node")
        workflow.add_edge("research_node", "execute_tools")
        workflow.add_conditional_edges(
            "execute_tools",
            self.route_task_progress,
            {"research_node": "research_node", "collect_informations": "collect_informations"},
        )
        workflow.add_edge("collect_informations", "evaluate_response")
        workflow.add_conditional_edges(
            "evaluate_response",
            self.route_evaluation,
            {"plan": "plan", "finalize_response": "finalize_response"},
        )
        workflow.add_edge("finalize_response", END)
        return workflow

    def route_scope(self, state: ResearchState):
        if state.get("scope_category") == "out_of_scope":
            return "out_of_scope_response"
        return "clarify_query"

    def route_clarification(self, state: ResearchState):
        if state.get("hitl_answer"):
            return "clarify_query"
        if state.get("final_response"):
            return "finalize_response"
        return "plan"

    def route_evaluation(self, state: ResearchState):
        if state.get("iterations", 0) >= 3:
            return "finalize_response"
        return "plan" if state.get("evaluation", {}).get("improvement_scopes") else "finalize_response"

    def route_task_progress(self, state: ResearchState):
        """Continue with the next planned task or synthesize all collected responses."""
        task_index = state.get("current_task_index", 0)
        task_count = len(state.get("tasks", []))
        return "research_node" if task_index < task_count else "collect_informations"

    def invoke(
        self,
        query: str,
        messages: list[Any] | None = None,
        hitl_answer: str = "",
        thread_id: str = "default",
        memory_context: dict[str, Any] | None = None,
    ):
        config = {"configurable": {"thread_id": thread_id}}
        if hitl_answer:
            result = self.graph.invoke(Command(resume=hitl_answer), config=config)
        else:
            result = self.graph.invoke(
                {
                    "query": query,
                    "messages": messages or [],
                    "memory_context": memory_context or {},
                },
                config=config,
            )
        if result.get("__interrupt__"):
            result["needs_hitl"] = True
            result["hitl_question"] = result["__interrupt__"][0].value["question"]
            result["final_response"] = result["hitl_question"]
        return result
