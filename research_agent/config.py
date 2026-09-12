import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _configured_hf_tokens() -> tuple[str, ...]:
    values = [
        os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip(),
        *[
            value
            for index in range(1, 6)
            for value in (
                os.getenv(f"HUGGINGFACEHUB_API_TOKEN{index}", "").strip(),
                os.getenv(f"HF_TOKEN{index}", "").strip(),
            )
        ],
    ]
    return tuple(dict.fromkeys(value for value in values if value and not value.startswith("your_")))


@dataclass(frozen=True)
class Settings:
    hf_token: str = os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
    hf_tokens: tuple[str, ...] = _configured_hf_tokens()
    llm_model_id: str = os.getenv("LLM_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
    embedding_model_id: str = os.getenv("EMBEDDING_MODEL_ID", "BAAI/bge-m3")
    embedding_cache_dir: str = os.getenv("EMBEDDING_CACHE_DIR", ".models")
    embedding_local_files_only: bool = os.getenv("EMBEDDING_LOCAL_FILES_ONLY", "false").lower() == "true"
    hf_provider: str = os.getenv("HF_PROVIDER", "auto")
    llm_max_new_tokens: int = int(os.getenv("LLM_MAX_NEW_TOKENS", "1024"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    llm_call_delay_seconds: int = int(os.getenv("LLM_CALL_DELAY_SECONDS", "20"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    chroma_dir: str = os.getenv("CHROMA_DIR", ".chroma")
    documents_dir: str = os.getenv("DOCUMENTS_DIR", "documents")
    rag_chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "900"))
    rag_chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "140"))
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "8"))
    max_research_iterations: int = int(os.getenv("MAX_RESEARCH_ITERATIONS", "2"))


settings = Settings()
