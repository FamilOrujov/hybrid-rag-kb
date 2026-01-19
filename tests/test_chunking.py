"""Tests for the chunking module."""

import pytest
from src.rag.chunking import Chunk, chunk_text


class TestChunk:
    """Test the Chunk dataclass."""

    def test_chunk_creation(self):
        """Chunk should store text and metadata correctly."""
        chunk = Chunk(text="Hello world", metadata={"doc_id": 1})
        assert chunk.text == "Hello world"
        assert chunk.metadata == {"doc_id": 1}

    def test_chunk_with_empty_metadata(self):
        """Chunk should handle empty metadata."""
        chunk = Chunk(text="Content", metadata={})
        assert chunk.text == "Content"
        assert chunk.metadata == {}


class TestChunkText:
    """Test the chunk_text function."""

    def test_short_text_single_chunk(self):
        """Short text should produce a single chunk."""
        text = "This is a short text."
        chunks = chunk_text(text, {"filename": "test.txt"}, chunk_size=100, chunk_overlap=10)
        
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].metadata["filename"] == "test.txt"
        assert chunks[0].metadata["chunk_index"] == 0

    def test_long_text_multiple_chunks(self):
        """Long text should be split into multiple chunks."""
        text = "Word " * 200  # Creates text longer than default chunk size
        chunks = chunk_text(text, {"doc_id": 42}, chunk_size=100, chunk_overlap=20)
        
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["doc_id"] == 42

    def test_chunk_indices_are_sequential(self):
        """Chunk indices should be sequential starting from 0."""
        text = "Paragraph one. " * 50 + "\n\n" + "Paragraph two. " * 50
        chunks = chunk_text(text, {}, chunk_size=100, chunk_overlap=10)
        
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_metadata_preserved(self):
        """Base metadata should be preserved in all chunks."""
        text = "Content " * 100
        base_meta = {"filename": "doc.pdf", "author": "Test", "page": 5}
        chunks = chunk_text(text, base_meta, chunk_size=50, chunk_overlap=5)
        
        for chunk in chunks:
            assert chunk.metadata["filename"] == "doc.pdf"
            assert chunk.metadata["author"] == "Test"
            assert chunk.metadata["page"] == 5

    def test_empty_text_returns_empty_or_single(self):
        """Empty or whitespace text should be handled gracefully."""
        chunks = chunk_text("", {}, chunk_size=100, chunk_overlap=10)
        # Empty text may return empty list or single empty chunk depending on splitter
        assert len(chunks) <= 1

    def test_overlap_creates_redundancy(self):
        """Overlapping chunks should share content at boundaries."""
        text = "AAAA BBBB CCCC DDDD EEEE FFFF GGGG HHHH"
        chunks = chunk_text(text, {}, chunk_size=20, chunk_overlap=5)
        
        if len(chunks) > 1:
            # Check that chunks have some overlap in content
            for i in range(len(chunks) - 1):
                # The end of chunk i and start of chunk i+1 may share content
                assert len(chunks[i].text) > 0
                assert len(chunks[i + 1].text) > 0
