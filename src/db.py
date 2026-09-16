import psycopg
from pgvector.psycopg import register_vector

from src.config import settings

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    title TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR({embedding_dim}) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""


def get_connection() -> psycopg.Connection:
    conn = psycopg.connect(settings.dsn, autocommit=True)
    register_vector(conn)
    return conn


def init_schema() -> None:
    # The `vector` extension may not exist yet on a fresh database, and
    # register_vector() requires it to already be registered in pg_type,
    # so run the schema-creation SQL over a plain connection first.
    with psycopg.connect(settings.dsn, autocommit=True) as conn:
        conn.execute(SCHEMA_SQL.format(embedding_dim=settings.embedding_dim))


def reset_schema() -> None:
    with get_connection() as conn:
        conn.execute("DROP TABLE IF EXISTS chunks CASCADE;")
        conn.execute("DROP TABLE IF EXISTS documents CASCADE;")
    init_schema()


if __name__ == "__main__":
    init_schema()
    print("Schema ready.")
