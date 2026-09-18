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


class ClarificationDecision(BaseModel):
    needs_clarification: bool = Field(description="Whether the research query needs a user clarification before planning")
    question: str = Field(description="One focused clarification question, or an empty string when clarification is not needed")
    final_query: str = Field(description="A clean, specific, expanded research query ready for planning when clarification is not needed")


class ResponseEvaluation(BaseModel):
    verdict: str = Field(description="Statement describing the quality and support of the response")
    improvement_scopes: list[str] = Field(
        default_factory=list,
        description="Specific areas needing improvement; empty when the response is ready",
    )


class ResearchDraft(BaseModel):
    answer: str = Field(description="The complete research answer with inline [source_id] citations.")
    citation_ids: list[str] = Field(
        default_factory=list,
        description="Source IDs that directly support claims in the answer, in citation order.",
    )


def llm_response_fixing_parser(model: type[BaseModel], llm):
    parser = PydanticOutputParser(pydantic_object=model)
    return OutputFixingParser.from_llm(
        parser=parser, llm=llm, max_retries=getattr(settings, "max_retries", 3)
    )
