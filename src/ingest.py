import sys
from pathlib import Path

from src.chunking import chunk_text
from src.db import get_connection, init_schema
from src.embeddings import embed_texts

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[tuple[str, str, str]]:
    """Returns list of (source_filename, title, full_text)."""
    docs = []
    for path in sorted(corpus_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip() if text.startswith("#") else path.stem
        docs.append((path.name, title, text))
    return docs


def ingest(corpus_dir: Path = CORPUS_DIR, reset: bool = False) -> None:
    from src.db import reset_schema

    if reset:
        reset_schema()
    else:
        init_schema()

    docs = load_corpus(corpus_dir)
    if not docs:
        print(f"No documents found in {corpus_dir}")
        return

    total_chunks = 0
    with get_connection() as conn:
        for source, title, text in docs:
            existing = conn.execute(
                "SELECT id FROM documents WHERE source = %s", (source,)
            ).fetchone()
            if existing:
                print(f"Skipping (already ingested): {source}")
                continue

            doc_id = conn.execute(
                "INSERT INTO documents (source, title) VALUES (%s, %s) RETURNING id",
                (source, title),
            ).fetchone()[0]

            chunks = chunk_text(text)
            if not chunks:
                continue

            embeddings = embed_texts([c.content for c in chunks])

            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (document_id, chunk_index, content, embedding) "
                    "VALUES (%s, %s, %s, %s)",
                    [
                        (doc_id, c.chunk_index, c.content, emb)
                        for c, emb in zip(chunks, embeddings)
                    ],
                )
            total_chunks += len(chunks)
            print(f"Ingested {source}: {len(chunks)} chunks")

    print(f"Done. {len(docs)} documents processed, {total_chunks} new chunks stored.")


if __name__ == "__main__":
    reset_flag = "--reset" in sys.argv
    ingest(reset=reset_flag)
