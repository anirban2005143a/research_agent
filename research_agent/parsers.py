from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.output_parsers import OutputFixingParser
from pydantic import BaseModel, Field

from .config import settings


class ResearchPlan(BaseModel):
    steps: list[str] = Field(description="Ordered research steps, each phrased as a focused question")


class ScopeDecision(BaseModel):
    category: str = Field(description="One of: out_of_scope, answerable, needs_research")
    reason: str = Field(description="Brief reason for the classification")
    response: str = Field(description="Natural response if out_of_scope; otherwise a brief acknowledgement")


class ClarificationDecision(BaseModel):
    needs_clarification: bool = Field(description="Whether the research query needs a user clarification before planning")
    question: str = Field(description="One focused clarification question, or an empty string when clarification is not needed")
    final_query: str = Field(description="A clean, specific, expanded research query ready for planning when clarification is not needed")


class ResponseEvaluation(BaseModel):
    score: int = Field(ge=1, le=10)
    issues: list[str] = Field(default_factory=list)
    needs_more_research: bool = False
    recommendation: str = ""


class ResearchDraft(BaseModel):
    answer: str = Field(description="The complete research answer with inline [source_id] citations.")
    citation_ids: list[str] = Field(
        default_factory=list,
        description="Source IDs that directly support claims in the answer, in citation order.",
    )


def fixing_parser(model: type[BaseModel], llm):
    parser = PydanticOutputParser(pydantic_object=model)
    return OutputFixingParser.from_llm(
        parser=parser, llm=llm, max_retries=getattr(settings, "max_retries", 3)
    )
