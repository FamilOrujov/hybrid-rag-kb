"""Tests for the FAISS vectorstore module."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from src.rag.vectorstore import FaissIndexManager, FaissSearchResult


class TestFaissSearchResult:
    """Test the FaissSearchResult dataclass."""

    def test_search_result_creation(self):
        """Should store ids and scores correctly."""
        result = FaissSearchResult(ids=[1, 2, 3], scores=[0.9, 0.8, 0.7])
        assert result.ids == [1, 2, 3]
        assert result.scores == [0.9, 0.8, 0.7]


class TestFaissIndexManager:
    """Test the FAISS index manager."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for FAISS index."""
        dir_path = tempfile.mkdtemp()
        yield dir_path
        shutil.rmtree(dir_path)

    @pytest.fixture
    def manager(self, temp_dir):
        """Create a FaissIndexManager instance."""
        return FaissIndexManager(temp_dir, use_gpu=False)

    def test_directory_created(self, temp_dir):
        """Manager should create directory if not exists."""
        new_dir = Path(temp_dir) / "nested" / "faiss"
        manager = FaissIndexManager(str(new_dir), use_gpu=False)
        assert new_dir.exists()

    def test_load_or_create_new_index(self, manager):
        """Should create new index when none exists."""
        manager.load_or_create(dim=128)
        
        assert manager.cpu_index is not None
        assert manager.dim == 128

    def test_add_vectors(self, manager):
        """Should add vectors to the index."""
        manager.load_or_create(dim=4)
        
        ids = [1, 2, 3]
        vectors = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32)
        
        manager.add(ids, vectors)
        
        assert manager.cpu_index.ntotal == 3

    def test_search_returns_results(self, manager):
        """Should return search results with ids and scores."""
        manager.load_or_create(dim=4)
        
        ids = [10, 20, 30]
        vectors = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32)
        manager.add(ids, vectors)
        
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        result = manager.search(query, k=2)
        
        assert isinstance(result, FaissSearchResult)
        assert len(result.ids) <= 2
        assert 10 in result.ids  # Closest match

    def test_search_similarity_ordering(self, manager):
        """Search should return most similar vectors first."""
        manager.load_or_create(dim=4)
        
        ids = [1, 2, 3]
        vectors = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)
        manager.add(ids, vectors)
        
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        result = manager.search(query, k=3)
        
        # ID 1 should be first (exact match), ID 2 second (similar), ID 3 last
        assert result.ids[0] == 1

    def test_persistence(self, temp_dir):
        """Index should persist to disk and reload."""
        # Create and save
        manager1 = FaissIndexManager(temp_dir, use_gpu=False)
        manager1.load_or_create(dim=4)
        
        vectors = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        manager1.add([42], vectors)
        
        # Load in new instance
        manager2 = FaissIndexManager(temp_dir, use_gpu=False)
        manager2.load_or_create(dim=4)
        
        assert manager2.cpu_index.ntotal == 1
        
        # Should find the vector
        result = manager2.search(vectors[0], k=1)
        assert 42 in result.ids

    def test_dimension_mismatch_raises(self, manager):
        """Should raise error when vector dimension mismatches index."""
        manager.load_or_create(dim=4)
        
        wrong_dim_vectors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)  # dim=3
        
        with pytest.raises(ValueError, match="vectors shape"):
            manager.add([1], wrong_dim_vectors)

    def test_empty_index_search(self, manager):
        """Searching empty index should return empty results."""
        manager.load_or_create(dim=4)
        
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        result = manager.search(query, k=5)
        
        assert result.ids == []
        assert result.scores == []

    def test_dtype_conversion(self, manager):
        """Should handle numpy dtype conversion."""
        manager.load_or_create(dim=4)
        
        # float64 input should be converted
        vectors = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        manager.add([1], vectors)
        
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        result = manager.search(query, k=1)
        
        assert 1 in result.ids
