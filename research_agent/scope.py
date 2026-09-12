RESEARCH_TERMS = {
    "research", "study", "paper", "papers", "evidence", "source", "sources", "compare",
    "analysis", "analyze", "literature", "review", "report", "investigate", "investigation",
    "citation", "cite", "fact check", "benchmark", "market", "academic", "scientific",
    "history", "policy", "technology", "trend", "data", "pros and cons", "state of the art",
}

RESEARCH_ONLY_REPLY = (
    "I am a research-oriented agent. I can help investigate a topic, compare evidence, "
    "summarize sources, analyze uploaded documents, and produce a cited research brief. "
    "Please ask a research question or request an evidence-based analysis."
)


def classify_query(query: str) -> tuple[bool, str]:
    normalized = query.lower().strip()
    if not normalized:
        return False, "Please provide a research question."
    if any(term in normalized for term in RESEARCH_TERMS):
        return True, "The request is research-oriented."
    return False, "The request does not ask for research, evidence, or analysis."
