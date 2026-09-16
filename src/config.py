from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Postgres / pgvector
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "ragdb"
    postgres_user: str = "raguser"
    postgres_password: str = "ragpassword"

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # Chunking
    chunk_size_tokens: int = 300
    chunk_overlap_tokens: int = 50

    # Hybrid retrieval
    retrieval_top_k: int = 5
    bm25_weight: float = 0.4
    vector_weight: float = 0.6

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ragas_judge_model: str = "gpt-oss:120b-cloud"

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
