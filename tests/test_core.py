"""Tests for the pieces that do not need a model download: chunking, BM25, the vector store, fusion and citations."""

import numpy as np
import pytest

from rag.bm25 import BM25, tokenize
from rag.chunking import ChunkerConfig, chunk_document, pack, split_blocks, split_sections
from rag.documents import Chunk, Document
from rag.generation import build_context, check_citations
from rag.index import Index
from rag.retrieval import Hit, Retriever, reciprocal_rank_fusion
from rag.vector_store import VectorStore

DOC = """# Security

Intro text about security.

## OAuth2 { #oauth2 }

Use OAuth2 with password flow.

```python
# this is a comment, not a heading
token = create_token()
```

### JWT

Sign tokens with a secret key.
"""


# Chunking

def test_sections_follow_heading_trail_and_ignore_hashes_in_code():
    sections = split_sections(DOC)
    trails = [" > ".join(t) for t, _ in sections]
    assert trails == ["Security", "Security > OAuth2", "Security > OAuth2 > JWT"]
    assert "# this is a comment" in sections[1][1]


def test_code_fence_stays_one_block():
    blocks = split_blocks("para one\n\n```python\na = 1\n\nb = 2\n```\n\npara two")
    assert blocks == ["para one", "```python\na = 1\n\nb = 2\n```", "para two"]


def test_pack_respects_max_chars_and_overlaps_short_tail():
    cfg = ChunkerConfig(max_chars=30, overlap_chars=10)
    chunks = pack(["a" * 15, "b" * 8, "c" * 15], cfg)
    assert all(len(c) <= 30 for c in chunks)
    assert chunks[1].startswith("b" * 8)  # the short tail was carried over


def test_oversized_block_is_split():
    cfg = ChunkerConfig(max_chars=50, overlap_chars=0, min_chars=1)
    text = "# T\n\n" + " ".join(["Sentence number one is here."] * 10)
    chunks = chunk_document(Document("t.md", text, "T"), cfg)
    assert len(chunks) > 1
    assert all(len(c.text) <= 50 for c in chunks)


# BM25

def test_tokenize_splits_snake_case_and_drops_stopwords():
    assert tokenize("The response_model") == ["response_model", "response", "model"]


def test_bm25_ranks_the_matching_document_first():
    bm = BM25().fit(["cats purr softly", "HTTPException returns a 404 status", "dogs bark"])
    assert bm.search("raise HTTPException 404")[0][0] == 1
    assert bm.search("nonexistentterm") == []


def test_bm25_prefers_rare_terms():
    bm = BM25().fit(["common common rare", "common common common", "common other"])
    assert bm.search("rare common")[0][0] == 0


# Vector store

def test_vector_store_cosine_top_k():
    store = VectorStore(2)
    store.add(np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32))
    results = store.search(np.array([1, 0.1]), k=2)
    assert [i for i, _ in results] == [0, 2]
    assert results[0][1] == pytest.approx(0.995, abs=1e-3)


def test_vector_store_round_trip(tmp_path):
    store = VectorStore(3)
    store.add(np.random.default_rng(0).normal(size=(5, 3)))
    store.save(tmp_path / "v.npy")
    loaded = VectorStore.load(tmp_path / "v.npy")
    np.testing.assert_allclose(loaded.vectors, store.vectors)


# Fusion and retrieval

def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]])
    assert {fused[0][0], fused[1][0]} == {1, 2}
    assert fused[-1][0] in (3, 4)


class KeywordEmbedder:
    """A deterministic stand in for a neural model: one dimension per vocabulary word."""

    name = "fake"
    vocab = ["cors", "origin", "upload", "file", "jwt", "token"]
    dim = len(vocab)

    def _vec(self, text):
        words = text.lower().split()
        return np.array([sum(w.startswith(v) for w in words) for v in self.vocab], dtype=np.float32) + 1e-3

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


def make_index():
    chunks = [
        Chunk(0, "cors.md", "CORS", "Allow another origin with CORSMiddleware"),
        Chunk(1, "files.md", "Files", "Upload a file with UploadFile"),
        Chunk(2, "jwt.md", "JWT", "Issue a JWT token after login"),
    ]
    emb = KeywordEmbedder()
    store = VectorStore(emb.dim)
    store.add(emb.embed_documents([c.text for c in chunks]))
    return Index(chunks, store, emb.name), emb


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_retriever_modes_find_the_right_chunk(mode):
    index, emb = make_index()
    hits = Retriever(index, emb).search("upload a file", k=2, mode=mode)
    assert hits[0].chunk.source == "files.md"
    assert hits[0].rank == 1


def test_index_save_and_load(tmp_path):
    index, emb = make_index()
    index.save(tmp_path)
    loaded = Index.load(tmp_path)
    assert [c.text for c in loaded.chunks] == [c.text for c in index.chunks]
    assert Retriever(loaded, emb).search("jwt token", k=1)[0].chunk.source == "jwt.md"


def test_retriever_rejects_mismatched_embedder():
    index, emb = make_index()
    index.embed_model = "other-model"
    with pytest.raises(ValueError):
        Retriever(index, emb)


# Prompt context and citations

def test_build_context_numbers_sources_and_respects_budget():
    hits = [Hit(Chunk(i, f"{i}.md", "S", "x" * 100), 1.0, i + 1) for i in range(5)]
    context, used = build_context(hits, max_chars=400)
    assert '<source id="1" file="0.md"' in context
    assert 1 <= len(used) < 5


def test_check_citations_flags_numbers_that_point_nowhere():
    valid, invalid = check_citations("Use CORSMiddleware [1][3]. Also see [7].", n_sources=3)
    assert valid == [1, 3]
    assert invalid == [7]
