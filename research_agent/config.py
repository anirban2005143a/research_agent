import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _get_env(name: str, default: str, *aliases: str) -> str:
    for candidate in (name, *aliases):
        value = os.getenv(candidate)
        if value is not None:
            return value
    return default


def _get_env_int(name: str, default: int, *aliases: str) -> int:
    for candidate in (name, *aliases):
        value = os.getenv(candidate)
        if value is not None and value != "":
            return int(value)
    return default


def _get_env_float(name: str, default: float, *aliases: str) -> float:
    for candidate in (name, *aliases):
        value = os.getenv(candidate)
        if value is not None and value != "":
            return float(value)
    return default


def _get_env_bool(name: str, default: bool, *aliases: str) -> bool:
    for candidate in (name, *aliases):
        value = os.getenv(candidate)
        if value is not None and value != "":
            return value.lower() == "true"
    return default


def _configured_hf_tokens() -> tuple[str, ...]:
    values = [
        os.getenv(f"HUGGINGFACEHUB_API_TOKEN{index}", "").strip()
        for index in range(1, 6)
    ]
    return tuple(dict.fromkeys(value for value in values if value and not value.startswith("your_")))


@dataclass(frozen=True)
class Settings:
    hf_tokens: tuple[str, ...] = _configured_hf_tokens()
    llm_model_id: str = _get_env("LLM_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
    embedding_model_id: str = _get_env("EMBEDDING_MODEL_ID", "BAAI/bge-m3")
    embedding_cache_dir: str = _get_env("EMBEDDING_CACHE_DIR", ".models")
    embedding_local_files_only: bool = _get_env_bool("EMBEDDING_LOCAL_FILES_ONLY", False)
    hf_provider: str = _get_env("HF_PROVIDER", "auto")
    llm_max_new_tokens: int = _get_env_int("LLM_MAX_NEW_TOKENS", 4096)
    llm_temperature: float = _get_env_float("LLM_TEMPERATURE", 0.1)
    llm_call_delay_seconds: int = _get_env_int("LLM_CALL_DELAY_SECONDS", 20)
    max_retries: int = _get_env_int("MAX_RETRIES", 3)
    retry_delay_seconds: float = _get_env_float("RETRY_DELAY_SECONDS", 1.0)
    session_memory_snapshot_dir: str = _get_env("SESSION_MEMORY_SNAPSHOT_DIR", "session_memory_snapshots")
    chroma_dir: str = _get_env("CHROMA_DIR", ".chroma")
    documents_dir: str = _get_env("DOCUMENTS_DIR", "documents")
    rag_chunk_size: int = _get_env_int("RAG_CHUNK_SIZE", 900)
    rag_chunk_overlap: int = _get_env_int("RAG_CHUNK_OVERLAP", 140)
    rag_embedding_batch_size: int = _get_env_int("RAG_EMBEDDING_BATCH_SIZE", 16)
    rag_top_k: int = _get_env_int("RAG_TOP_K", 8)
    rag_candidate_multiplier: int = _get_env_int("RAG_CANDIDATE_MULTIPLIER", 6)
    rag_cross_encoder_model_id: str = _get_env("CROSS_ENCODER_MODEL_ID", "BAAI/bge-reranker-v2-m3")
    rag_cross_encoder_enabled: bool = _get_env_bool("CROSS_ENCODER_ENABLED", True)
    rag_cross_encoder_batch_size: int = _get_env_int("CROSS_ENCODER_BATCH_SIZE", 8)
    rag_cross_encoder_local_files_only: bool = _get_env_bool("CROSS_ENCODER_LOCAL_FILES_ONLY", False)
    rag_dense_weight: float = _get_env_float("RAG_DENSE_WEIGHT", 0.35)
    rag_bm25_weight: float = _get_env_float("RAG_BM25_WEIGHT", 0.25)
    rag_cross_encoder_weight: float = _get_env_float("RAG_CROSS_ENCODER_WEIGHT", 0.40, "CROSS_ENCODER_WEIGHT")
    rag_cross_encoder_cache_dir: str = _get_env("CROSS_ENCODER_CACHE_DIR", ".models")
    max_research_iterations: int = _get_env_int("MAX_RESEARCH_ITERATIONS", 2)


settings = Settings()
