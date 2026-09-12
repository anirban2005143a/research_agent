from threading import Lock

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import Runnable

from .config import settings


class RotatingHuggingFaceChat(Runnable):
    """Create each hosted chat client with the next configured token."""

    def __init__(self, tokens: tuple[str, ...], tools=None, **bind_kwargs):
        self.tokens = tokens
        self.tools = tools
        self.bind_kwargs = bind_kwargs
        self._index = 0
        self._lock = Lock()

    def _next_token(self) -> str:
        with self._lock:
            token = self.tokens[self._index % len(self.tokens)]
            self._index += 1
        return token

    def invoke(self, input, config=None, **kwargs):
        endpoint = HuggingFaceEndpoint(
            repo_id=settings.llm_model_id,
            huggingfacehub_api_token=self._next_token(),
            max_new_tokens=settings.llm_max_new_tokens,
            temperature=settings.llm_temperature,
        )
        model = ChatHuggingFace(llm=endpoint)
        if self.tools:
            model = model.bind_tools(self.tools)
        if self.bind_kwargs:
            model = model.bind(**self.bind_kwargs)
        return model.invoke(input, config=config, **kwargs)

    def bind_tools(self, tools, **kwargs):
        return RotatingHuggingFaceChat(
            self.tokens,
            tools=tools,
            **self.bind_kwargs,
            **kwargs,
        )


def build_llm():
    tokens = settings.hf_tokens or ((settings.hf_token,) if settings.hf_token and not settings.hf_token.startswith("your_") else ())
    if not tokens:
        raise RuntimeError("Configure HUGGINGFACEHUB_API_TOKEN or HF_TOKEN1..HF_TOKEN5 in .env.")
    return RotatingHuggingFaceChat(tokens)
