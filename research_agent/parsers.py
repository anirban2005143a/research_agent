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


class QualityReview(BaseModel):
    score: int = Field(ge=1, le=10)
    issues: list[str] = Field(default_factory=list)
    needs_more_research: bool = False
    recommendation: str = ""


def fixing_parser(model: type[BaseModel], llm):
    parser = PydanticOutputParser(pydantic_object=model)
    return OutputFixingParser.from_llm(parser=parser, llm=llm, max_retries=settings.max_retries)
