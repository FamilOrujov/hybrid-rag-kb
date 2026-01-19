"""Pytest configuration and shared fixtures."""

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_directory():
    """
    Create a temporary directory for tests that need filesystem access.
    Automatically cleaned up after test completes.
    """
    dir_path = tempfile.mkdtemp()
    yield Path(dir_path)
    shutil.rmtree(dir_path, ignore_errors=True)


@pytest.fixture
def sample_text():
    """Sample text for chunking and processing tests."""
    return """
    Artificial intelligence has made remarkable progress in recent years.
    Machine learning models can now understand and generate human language
    with unprecedented accuracy.

    Deep learning architectures like transformers have revolutionized
    natural language processing. These models learn patterns from vast
    amounts of text data.

    Retrieval augmented generation combines the power of large language
    models with external knowledge bases. This approach grounds AI responses
    in factual information.
    """


@pytest.fixture
def sample_chunks():
    """Sample chunk data for fusion and retrieval tests."""
    return [
        {
            "chunk_id": 1,
            "text": "Machine learning is a subset of artificial intelligence.",
            "filename": "ml_basics.pdf",
            "metadata": {"page": 1, "section": "intro"},
        },
        {
            "chunk_id": 2,
            "text": "Neural networks are inspired by biological neurons.",
            "filename": "ml_basics.pdf",
            "metadata": {"page": 2, "section": "foundations"},
        },
        {
            "chunk_id": 3,
            "text": "Transformers use attention mechanisms for sequence modeling.",
            "filename": "transformers.pdf",
            "metadata": {"page": 1, "section": "architecture"},
        },
    ]


@pytest.fixture
def sample_answer_with_citations():
    """Sample LLM answer with proper citations."""
    return """
    Machine learning represents a fundamental shift in how we approach
    problem solving with computers [Source: ml_basics.pdf | cid:1].

    The architecture of neural networks draws inspiration from biological
    systems, specifically the way neurons communicate [Source: ml_basics.pdf | cid:2].

    Modern language models rely heavily on transformer architectures,
    which use self-attention to process sequences [Source: transformers.pdf | cid:3].
    """
