"""A flat vector index in numpy.

Vectors are L2 normalised when added, so cosine similarity is a single matrix
product. Exact search over a few thousand chunks takes well under a
millisecond; an approximate index (HNSW, IVF) only pays off at far larger scale.
"""

from __future__ import annotations

import numpy as np


def normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.clip(norms, 1e-12, None)


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return self.vectors.shape[0]

    def add(self, vectors: np.ndarray) -> None:
        vectors = normalize(vectors)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected shape (n, {self.dim}), got {vectors.shape}")
        self.vectors = np.vstack([self.vectors, vectors])

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        if len(self) == 0:
            return []
        q = normalize(query.reshape(1, -1))[0]
        sims = self.vectors @ q
        k = min(k, len(sims))
        # argpartition finds the top k in linear time, then only those k get sorted.
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top], kind="stable")]
        return [(int(i), float(sims[i])) for i in top]

    def save(self, path) -> None:
        np.save(path, self.vectors)

    @classmethod
    def load(cls, path) -> "VectorStore":
        vectors = np.load(path)
        store = cls(vectors.shape[1])
        store.vectors = vectors.astype(np.float32)
        return store
