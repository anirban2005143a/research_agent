from collections.abc import Callable
import re

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from .utils import retry_call


class MemoryExtraction(BaseModel):
    user_info: list[str] = Field(
        default_factory=list,
        description="Very few durable facts about the user that help with future responses.",
    )
    session_context: list[str] = Field(
        default_factory=list,
        description="Very few durable facts about the research context, prior findings, and useful constraints.",
    )


class ShortTermMemory:
    """Stores only a compact RAM snapshot for one session: user facts and session context."""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id
        self.user_info: list[str] = []
        self.session_context: list[str] = []

    def _merge_points(self, existing: list[str], new_points: list[str], limit: int = 5) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for item in [*existing, *new_points]:
            cleaned = " ".join(str(item).split()).strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(cleaned)
            if len(merged) >= limit:
                break
        return merged

    def update_from_query(self, query: str, llm) -> None:
        if not query or not llm:
            return
        prompt = (
            "From the user's new query, extract only a few durable facts that are useful for future responses. "
            "Keep it to a short list of strings. Do not store the raw query itself. "
            "Only include facts like name, role, preferences, constraints, domain, goals, or important context.\n\n"
            f"Existing user info: {self.user_info}\n"
            f"New query: {query}\n\n"
            "Return only the requested structured format. Keep the list brief and deduplicated."
        )
        try:
            parser = PydanticOutputParser(pydantic_object=MemoryExtraction)
            result = retry_call(
                lambda: llm.invoke([HumanMessage(content=f"{prompt}\n\n{parser.get_format_instructions()}")]),
                "user_info_extraction_llm",
            )
            extraction = parser.parse(result.content)
            self.user_info = self._merge_points(self.user_info, extraction.user_info, limit=5)
        except Exception:
            pass

    def update_from_response(self, response: str, llm) -> None:
        if not response or not llm:
            return
        prompt = (
            "From the final answer, extract only a few useful durable facts about the research session. "
            "Keep it to a short list of strings. Do not store the raw answer itself. "
            "Only include useful context such as the topic, key conclusion, constraints, unresolved question, or important reminder.\n\n"
            f"Existing session context: {self.session_context}\n"
            f"New final answer: {response}\n\n"
            "Return only the requested structured format. Keep the list brief and deduplicated."
        )
        try:
            parser = PydanticOutputParser(pydantic_object=MemoryExtraction)
            result = retry_call(
                lambda: llm.invoke([HumanMessage(content=f"{prompt}\n\n{parser.get_format_instructions()}")]),
                "session_context_extraction_llm",
            )
            extraction = parser.parse(result.content)
            self.session_context = self._merge_points(self.session_context, extraction.session_context, limit=5)
        except Exception:
            pass

    def context(self) -> dict:
        return {
            "user_info": self.user_info,
            "session_context": self.session_context,
        }


class SessionMemoryStore:
    """Keep per-session RAM memory keyed by session_id so different sessions never share data."""

    def __init__(self):
        self._sessions: dict[str, ShortTermMemory] = {}

    def get_or_create(self, session_id: str) -> ShortTermMemory:
        if not session_id:
            raise ValueError("session_id is required to create or fetch session memory")
        memory = self._sessions.get(session_id)
        if memory is None:
            memory = ShortTermMemory(session_id=session_id)
            self._sessions[session_id] = memory
        return memory

    def get(self, session_id: str) -> ShortTermMemory | None:
        return self._sessions.get(session_id)

    def remove(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def clear(self) -> None:
        self._sessions.clear()


SESSION_MEMORY_STORE = SessionMemoryStore()
