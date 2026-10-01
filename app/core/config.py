from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://rag_user:rag_pass@localhost:5432/ragdb"

    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB
    default_chunk_size: int = 500  # in "tokens" -- see ingestion/chunking.py
    default_chunk_overlap: int = 50

    # Day 29: embeddings, generated LOCALLY via fastembed (ONNX Runtime) --
    # no API key, no network call at request time, no provider outage to
    # retry against. BAAI/bge-small is a small (~130MB), fast,
    # well-regarded general-purpose embedding model.
    #
    # Trade-off vs a hosted embedding API: no cost and no external
    # dependency at request time, at the price of somewhat lower embedding
    # quality than a large hosted model, and a one-time model download on
    # first use (cached afterward -- see the fastembed_cache volume in
    # docker-compose.yml).
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Must match the Vector(...) dimension in models.py AND the migration
    # that creates the pgvector column -- all three have to agree.
    embedding_dimensions: int = 384
    embedding_batch_size: int = 64

    # Day 31: generation. Embeddings run locally (Groq has no embeddings
    # endpoint -- confirmed the hard way in this project's Day 29 build),
    # but a fluent grounded answer needs a real chat model, so this talks
    # to an OpenAI-compatible chat API. Groq's chat endpoint is real and
    # free-tier friendly, which is why it's the default here.
    llm_api_key: str = "sk-placeholder-set-in-env"
    llm_base_url: str | None = "https://api.groq.com/openai/v1"
    llm_model: str = "llama-3.1-8b-instant"
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2
    llm_retry_backoff_seconds: float = 0.5

    @field_validator("llm_base_url", mode="before")
    @classmethod
    def empty_string_means_unset(cls, v):
        """
        An env var set to LLM_BASE_URL= (nothing after the =) arrives here
        as an empty string, not "unset" -- normalizing "" to None means a
        blank env var falls back to OpenAI's default endpoint instead of
        failing later with a confusing "missing protocol" error.
        """
        if v == "":
            return None
        return v

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
