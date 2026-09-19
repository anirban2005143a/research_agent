from enum import Enum

from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.output_parsers import OutputFixingParser
from pydantic import BaseModel, Field

from .config import settings


class ScopeCategory(str, Enum):
    OUT_OF_SCOPE = "out_of_scope"
    IN_SCOPE = "in_scope"


class ScopeDecision(BaseModel):
    category: ScopeCategory


class ResearchPlan(BaseModel):
    tasks: list[str] = Field(description="Ordered research tasks to execute one at a time")


class ResearchToolCall(BaseModel):
    name: str = Field(description="Exact name of one available research tool")
    arguments: dict[str, object] = Field(default_factory=dict, description="Arguments for the selected tool")


class ResearchToolSelection(BaseModel):
    tool_calls: list[ResearchToolCall] = Field(
        default_factory=list,
        description="All tool calls needed for the current research task",
    )


class ClarificationDecision(BaseModel):
    needs_clarification: bool = Field(description="Whether the research query needs a user clarification before planning")
    question: str = Field(description="One focused clarification question, or an empty string when clarification is not needed")
    final_query: str = Field(description="A clean, specific, expanded research query ready for planning when clarification is not needed")


class ResponseEvaluation(BaseModel):
    verdict: str = Field(description="Statement describing the quality and support of the response")
    needs_improvement: bool = Field(
        default=False,
        description="True only when a material evidence, correctness, citation, or completeness problem requires another research pass.",
    )
    improvement_scopes: list[str] = Field(
        default_factory=list,
        description="Specific areas needing improvement; empty when the response is ready",
    )


class ResearchDraft(BaseModel):
    answer: str = Field(description="The complete research answer with inline citations where appropriate.")
    citations: list[str] = Field(
        default_factory=list,
        description="Source names, URLs, filenames, or other exact citation strings used in the answer.",
    )


class CitationMerge(BaseModel):
    citations: list[str] = Field(
        default_factory=list,
        description="Necessary, unique citation strings selected from the old and recent citation lists.",
    )


def llm_response_fixing_parser(model: type[BaseModel], llm):
    parser = PydanticOutputParser(pydantic_object=model)
    return OutputFixingParser.from_llm(
        parser=parser, llm=llm, max_retries=getattr(settings, "max_retries", 3)
    )
