from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.output_parsers import OutputFixingParser
from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    steps: list[str] = Field(description="Ordered research steps, each phrased as a focused question")


class ThinkingDecision(BaseModel):
    needs_planner: bool = Field(description="Whether the request needs a multi-step research plan")
    summary: str = Field(description="A short user-safe description of the research approach")


class QualityReview(BaseModel):
    score: int = Field(ge=1, le=10)
    issues: list[str] = Field(default_factory=list)
    needs_more_research: bool = False
    recommendation: str = ""


def fixing_parser(model: type[BaseModel], llm):
    parser = PydanticOutputParser(pydantic_object=model)
    return OutputFixingParser.from_llm(parser=parser, llm=llm, max_retries=3)
