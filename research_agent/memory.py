from collections.abc import Callable
import re

from langchain_core.messages import HumanMessage


class ShortTermMemory:
    """Keeps recent turns verbatim and compresses older turns into a preference/context note."""

    def __init__(self, recent_limit: int = 5):
        self.recent_limit = recent_limit
        self.messages: list[dict[str, str]] = []
        self.summary = ""
        self.preferences: dict[str, str] = {}

    def add(self, role: str, content: str, summarizer: Callable | None = None):
        self.messages.append({"role": role, "content": content})
        self._capture_preferences(content)
        if len(self.messages) > self.recent_limit:
            older = self.messages[: -self.recent_limit]
            self.messages = self.messages[-self.recent_limit :]
            transcript = "\n".join(
                f"{item['role']}: {item['content']}" for item in older
            )
            self.summary = self._summarize(transcript, summarizer)

    def _summarize(self, transcript: str, summarizer: Callable | None) -> str:
        if summarizer:
            try:
                return str(
                    summarizer(
                        [
                            HumanMessage(
                                content=(
                                    "Summarize durable user preferences and research context. "
                                    "Keep only useful facts.\n\n" + transcript
                                )
                            )
                        ]
                    ).content
                )
            except Exception:
                pass
        return (self.summary + " " + transcript)[-2000:]

    def _capture_preferences(self, content: str) -> None:
        preference_patterns = {
            "response_style": r"(?:prefer|preference is|please use)\s+(.*?)(?:[.!?]|$)",
            "citation_style": r"(?:cite|citation)\s+(.*?)(?:[.!?]|$)",
        }
        lowered = content.lower()
        for key, pattern in preference_patterns.items():
            match = re.search(pattern, lowered)
            if match:
                self.preferences[key] = match.group(1).strip()

    def context(self) -> dict:
        return {
            "summary": self.summary,
            "recent_messages": self.messages,
            "preferences": self.preferences,
        }
