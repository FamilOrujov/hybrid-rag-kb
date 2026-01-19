from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass
class FaissSearchResult:
    ids: list[int]
    scores: list[float]


class FaissIndexManager:
    """
    Manages a persistent FAISS index on disk.

    Design choice:
    - We keep the "source of truth" as a CPU index on disk (write_index/read_index).
    - For fast searching, we clone the CPU index to GPU when available.

    Why IndexIDMap2.
    - It lets us attach our own integer IDs to vectors via add_with_ids.
      That makes FAISS IDs equal to SQLite chunk IDs. :contentReference[oaicite:0]{index=0}
    """

    def __init__(self, dir_path: str, use_gpu: bool, gpu_device: int = 0):
        self.dir = Path(dir_path)
        self.dir.mkdir(parents=True, exist_ok=True)

        self.use_gpu = use_gpu
        self.gpu_device = gpu_device

        self.cpu_index: faiss.Index | None = None
        self.gpu_index: faiss.Index | None = None
        self.dim: int | None = None

        # Correct, consistent attribute name
        self.index_file = self.dir / "index.faiss"

    def _build_cpu_index(self, dim: int) -> faiss.Index:
        base = faiss.IndexFlatIP(dim)
        return faiss.IndexIDMap2(base)

    def load_or_create(self, dim: int) -> None:
        """
        If index exists, load it.
        Otherwise create a new one with the provided embedding dimension.
        """
        if self.index_file.exists():
            self.cpu_index = faiss.read_index(str(self.index_file))
            self.dim = int(self.cpu_index.d)
        else:
            self.cpu_index = self._build_cpu_index(dim)
            self.dim = dim
            self._save_cpu()

        self._refresh_gpu_copy()

    def _save_cpu(self) -> None:
        assert self.cpu_index is not None
        faiss.write_index(self.cpu_index, str(self.index_file))

    def _refresh_gpu_copy(self) -> None:
        self.gpu_index = None

        if not self.use_gpu:
            return

        if not hasattr(faiss, "get_num_gpus"):
            return

        if faiss.get_num_gpus() <= 0:
            return

        assert self.cpu_index is not None

        try:
            res = faiss.StandardGpuResources()
            self.gpu_index = faiss.index_cpu_to_gpu(res, self.gpu_device, self.cpu_index)
        except RuntimeError as e:
            # Handle CUDA out of memory or other GPU errors gracefully
            # Fall back to CPU-only mode
            self.gpu_index = None
            import logging

            logging.warning(f"Failed to copy FAISS index to GPU, falling back to CPU: {e}")

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        """
        ids: SQLite chunk ids
        vectors: float32 array shape (n, dim)
        """
        assert self.cpu_index is not None
        assert self.dim is not None

        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"vectors shape must be (n, {self.dim}), got {vectors.shape}")

        # Normalize so inner product behaves like cosine similarity
        faiss.normalize_L2(vectors)

        ids64 = np.array(ids, dtype=np.int64)
        self.cpu_index.add_with_ids(vectors, ids64)

        self._save_cpu()
        self._refresh_gpu_copy()

    def search(self, query_vector: np.ndarray, k: int) -> FaissSearchResult:
        assert self.cpu_index is not None
        assert self.dim is not None

        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        if query_vector.shape[1] != self.dim:
            raise ValueError(f"query_vector dim must be {self.dim}, got {query_vector.shape[1]}")

        faiss.normalize_L2(query_vector)

        index = self.gpu_index if self.gpu_index is not None else self.cpu_index
        scores, ids = index.search(query_vector, k)

        ids_list = [int(x) for x in ids[0] if int(x) != -1]
        scores_list = [float(x) for x in scores[0][: len(ids_list)]]
        return FaissSearchResult(ids=ids_list, scores=scores_list)
