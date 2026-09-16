from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings


@dataclass
class Chunk:
    content: str
    chunk_index: int


def chunk_text(text: str) -> list[Chunk]:
    """Split text into overlapping chunks along paragraph/sentence boundaries
    where possible, falling back to smaller separators only when needed."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=settings.chunk_size_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = splitter.split_text(text)
    return [Chunk(content=piece.strip(), chunk_index=i) for i, piece in enumerate(pieces) if piece.strip()]
