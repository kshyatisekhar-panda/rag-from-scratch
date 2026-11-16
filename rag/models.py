"""Local models: a bi-encoder for embeddings and a cross-encoder for reranking.

Both run on CPU and are downloaded once from Hugging Face on first use.
sentence-transformers is imported lazily so the rest of the package (and the
tests) work without loading torch.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# BGE models were trained with this instruction on the query side only.
QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, name: str = DEFAULT_EMBED_MODEL, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.batch_size = batch_size
        self.model = SentenceTransformer(name)
        # Renamed in sentence-transformers 6; keep working on older releases too.
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()
        self.query_prefix = QUERY_PREFIXES.get(name, "")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=len(texts) > 64
        ).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode(self.query_prefix + text, normalize_embeddings=True).astype(np.float32)


class CrossEncoderReranker:
    """Scores (query, passage) pairs jointly. Slower than a bi-encoder, so it only sees the top candidates."""

    def __init__(self, name: str = DEFAULT_RERANK_MODEL):
        from sentence_transformers import CrossEncoder

        self.name = name
        self.model = CrossEncoder(name)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self.model.predict([(query, p) for p in passages])]
