"""Tests for the hybrid fusion (RRF) module."""

import pytest
from src.rag.hybrid_fusion import rrf_fuse


def make_result(chunk_id: int, text: str = "", filename: str = "test.pdf", **kwargs):
    """Helper to create a result dict."""
    return {
        "chunk_id": chunk_id,
        "text": text or f"Content for chunk {chunk_id}",
        "filename": filename,
        "metadata": kwargs.get("metadata", {}),
        **kwargs,
    }


class TestRRFFuse:
    """Test Reciprocal Rank Fusion implementation."""

    def test_empty_inputs_returns_empty(self):
        """Empty inputs should return empty list."""
        result = rrf_fuse([], [])
        assert result == []

    def test_bm25_only(self):
        """Should work with only BM25 results."""
        bm25 = [make_result(1), make_result(2), make_result(3)]
        result = rrf_fuse(bm25, [])
        
        assert len(result) == 3
        assert result[0]["chunk_id"] == 1  # First rank has highest score
        assert result[1]["chunk_id"] == 2
        assert result[2]["chunk_id"] == 3

    def test_vector_only(self):
        """Should work with only vector results."""
        vec = [make_result(10), make_result(20), make_result(30)]
        result = rrf_fuse([], vec)
        
        assert len(result) == 3
        assert result[0]["chunk_id"] == 10

    def test_fusion_combines_rankings(self):
        """Fusion should combine rankings from both sources."""
        bm25 = [make_result(1), make_result(2), make_result(3)]
        vec = [make_result(2), make_result(3), make_result(4)]
        
        result = rrf_fuse(bm25, vec)
        
        # Chunk 2 appears in both at good ranks, should be boosted
        chunk_ids = [r["chunk_id"] for r in result]
        assert 2 in chunk_ids
        assert 3 in chunk_ids

    def test_duplicate_chunks_merged(self):
        """Same chunk from both sources should be merged, not duplicated."""
        bm25 = [make_result(1), make_result(2)]
        vec = [make_result(1), make_result(3)]
        
        result = rrf_fuse(bm25, vec)
        
        chunk_ids = [r["chunk_id"] for r in result]
        assert chunk_ids.count(1) == 1  # Chunk 1 appears only once

    def test_fused_score_calculation(self):
        """Fused score should follow RRF formula."""
        bm25 = [make_result(1)]  # rank 1
        vec = [make_result(1)]   # rank 1
        
        result = rrf_fuse(bm25, vec, rrf_k=60, w_bm25=1.0, w_vec=1.0)
        
        # Expected: 1/(60+1) + 1/(60+1) = 2/61
        expected_score = 2 / 61
        assert abs(result[0]["fused_score"] - expected_score) < 0.0001

    def test_top_k_limits_results(self):
        """Should return at most top_k results."""
        bm25 = [make_result(i) for i in range(20)]
        vec = [make_result(i + 20) for i in range(20)]
        
        result = rrf_fuse(bm25, vec, top_k=5)
        assert len(result) == 5

    def test_weights_affect_ranking(self):
        """Different weights should affect final ranking."""
        bm25 = [make_result(1)]  # Only in BM25
        vec = [make_result(2)]   # Only in vector
        
        # Heavy BM25 weight
        result_bm25_heavy = rrf_fuse(bm25, vec, w_bm25=10.0, w_vec=1.0)
        assert result_bm25_heavy[0]["chunk_id"] == 1
        
        # Heavy vector weight
        result_vec_heavy = rrf_fuse(bm25, vec, w_bm25=1.0, w_vec=10.0)
        assert result_vec_heavy[0]["chunk_id"] == 2

    def test_results_sorted_by_fused_score(self):
        """Results should be sorted by fused_score descending."""
        bm25 = [make_result(i) for i in range(10)]
        vec = [make_result(i) for i in range(5, 15)]
        
        result = rrf_fuse(bm25, vec)
        
        scores = [r["fused_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_metadata_preserved(self):
        """Metadata from original results should be preserved."""
        bm25 = [make_result(1, metadata={"page": 5, "section": "intro"})]
        
        result = rrf_fuse(bm25, [])
        
        assert result[0]["metadata"]["page"] == 5
        assert result[0]["metadata"]["section"] == "intro"

    def test_scores_from_sources_preserved(self):
        """Original BM25 and vector scores should be preserved."""
        bm25 = [{"chunk_id": 1, "text": "t", "filename": "f", "bm25_score": 0.95}]
        vec = [{"chunk_id": 1, "text": "t", "filename": "f", "vec_score": 0.88}]
        
        result = rrf_fuse(bm25, vec)
        
        assert result[0]["bm25_score"] == 0.95
        assert result[0]["vec_score"] == 0.88
