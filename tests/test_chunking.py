from src.chunking import chunk_text


def test_chunk_text_splits_long_document():
    text = "Paragraph one.\n\n" + ("word " * 800)
    chunks = chunk_text(text)
    assert len(chunks) > 1
    for c in chunks:
        assert c.content.strip() != ""


def test_chunk_text_indices_are_sequential():
    text = "word " * 800
    chunks = chunk_text(text)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_text_empty_string():
    assert chunk_text("") == []
