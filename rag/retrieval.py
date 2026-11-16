"""Dense, keyword and hybrid retrieval, with optional cross-encoder reranking.

Hybrid search uses Reciprocal Rank Fusion:
    rrf(d) = sum over rankers r of 1 / (k + rank_r(d))
It only looks at ranks, so the two very different score scales (cosine in
[-1, 1], BM25 unbounded) never have to be calibrated against each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .chunking import searchable_text
from .documents import Chunk
from .index import Index
from .models import CrossEncoderReranker, Embedder

Mode = Literal["dense", "bm25", "hybrid"]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


class Retriever:
    def __init__(
        self,
        index: Index,
        embedder: Embedder,
        reranker: CrossEncoderReranker | None = None,
        candidates: int = 50,
        rrf_k: int = 60,
    ):
        if embedder.name != index.embed_model:
            raise ValueError(f"Index was built with {index.embed_model} but the embedder is {embedder.name}")
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
        self.candidates = candidates
        self.rrf_k = rrf_k

    def _dense(self, query: str, n: int) -> list[tuple[int, float]]:
        return self.index.vectors.search(self.embedder.embed_query(query), n)

    def _bm25(self, query: str, n: int) -> list[tuple[int, float]]:
        return self.index.bm25.search(query, n)

    def search(self, query: str, k: int = 5, mode: Mode = "hybrid", rerank: bool | None = None) -> list[Hit]:
        rerank = self.reranker is not None if rerank is None else rerank
        n = max(self.candidates, k)
        if mode == "dense":
            ranked = self._dense(query, n)
        elif mode == "bm25":
            ranked = self._bm25(query, n)
        elif mode == "hybrid":
            rankings = [[i for i, _ in self._dense(query, n)], [i for i, _ in self._bm25(query, n)]]
            ranked = reciprocal_rank_fusion(rankings, self.rrf_k)[:n]
        else:
            raise ValueError(f"Unknown mode {mode!r}")

        if rerank:
            if self.reranker is None:
                raise ValueError("rerank=True but no reranker was loaded")
            pool = [i for i, _ in ranked[: self.candidates]]
            scores = self.reranker.score(query, [searchable_text(self.index.chunks[i]) for i in pool])
            ranked = sorted(zip(pool, scores), key=lambda kv: -kv[1])

        return [Hit(self.index.chunks[i], s, r) for r, (i, s) in enumerate(ranked[:k], start=1)]
