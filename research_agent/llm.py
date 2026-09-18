import time
from threading import Lock

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import Runnable

from .config import settings
from .utils import log


class LLMModule(Runnable):
    """Create each hosted chat client with the next configured token."""

    def __init__(self, tokens: tuple[str, ...] | None = None, tools=None, **bind_kwargs):
        self.tokens = tokens if tokens is not None else settings.hf_tokens
        if not self.tokens:
            raise RuntimeError("Configure HUGGINGFACEHUB_API_TOKEN1..HUGGINGFACEHUB_API_TOKEN5 in .env.")
        self.tools = tools
        self.bind_kwargs = bind_kwargs
        self._index = 0
        self._lock = Lock()

    def _next_token(self) -> str:
        with self._lock:
            token = self.tokens[self._index % len(self.tokens)]
            self._index += 1
        return token

    def _build_model(self, token: str):
        endpoint = HuggingFaceEndpoint(
            repo_id=getattr(settings, "llm_model_id", "meta-llama/Llama-3.1-8B-Instruct"),
            huggingfacehub_api_token=token,
            max_new_tokens=getattr(settings, "llm_max_new_tokens", 1024),
            temperature=getattr(settings, "llm_temperature", 0.1),
        )
        model = ChatHuggingFace(llm=endpoint)
        if self.tools:
            model = model.bind_tools(self.tools)
        if self.bind_kwargs:
            model = model.bind(**self.bind_kwargs)
        return model

    def invoke(self, input, config=None, **kwargs):
        delay_seconds = getattr(settings, "llm_call_delay_seconds", 20)
        log(f"llm.throttle | delay_seconds={delay_seconds}")
        time.sleep(delay_seconds)
        model = self._build_model(self._next_token())
        return model.invoke(input, config=config, **kwargs)

    def bind_tools(self, tools, **kwargs):
        return LLMModule(
            self.tokens,
            tools=tools,
            **self.bind_kwargs,
            **kwargs,
        )


def build_llm():
    configured_tokens = getattr(settings, "hf_tokens", ())
    if not configured_tokens:
        raise RuntimeError("Configure HUGGINGFACEHUB_API_TOKEN1..HUGGINGFACEHUB_API_TOKEN5 in .env.")
    return LLMModule(configured_tokens)
