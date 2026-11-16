"""Building, saving and loading the search index.

On disk an index is a folder with three files:
    chunks.jsonl     one chunk per line (text, source file, heading trail)
    vectors.npy      the embedding matrix, row i belongs to chunk i
    manifest.json    embedding model, chunker settings, counts

BM25 is rebuilt from chunks.jsonl at load time. It takes milliseconds, so storing it would only add a format to keep in sync.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from .bm25 import BM25
from .chunking import ChunkerConfig, chunk_documents, searchable_text
from .documents import Chunk, load_directory
from .models import Embedder
from .vector_store import VectorStore


class Index:
    def __init__(self, chunks: list[Chunk], vectors: VectorStore, embed_model: str, manifest: dict | None = None):
        self.chunks = chunks
        self.vectors = vectors
        self.embed_model = embed_model
        self.manifest = manifest or {}
        self.bm25 = BM25().fit([searchable_text(c) for c in chunks])

    @classmethod
    def build(cls, corpus_dir: str | Path, embedder: Embedder, cfg: ChunkerConfig | None = None) -> "Index":
        cfg = cfg or ChunkerConfig()
        started = time.perf_counter()
        docs = load_directory(corpus_dir)
        chunks = chunk_documents(docs, cfg)
        if not chunks:
            raise ValueError(f"No text found in {corpus_dir}")
        store = VectorStore(embedder.dim)
        store.add(embedder.embed_documents([searchable_text(c) for c in chunks]))
        manifest = {
            "corpus": str(Path(corpus_dir).as_posix()),
            "documents": len(docs),
            "chunks": len(chunks),
            "embed_model": embedder.name,
            "dim": embedder.dim,
            "chunker": asdict(cfg),
            "build_seconds": round(time.perf_counter() - started, 1),
        }
        return cls(chunks, store, embedder.name, manifest)

    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        self.vectors.save(folder / "vectors.npy")
        manifest = {**self.manifest, "embed_model": self.embed_model, "chunks": len(self.chunks)}
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, folder: str | Path) -> "Index":
        folder = Path(folder)
        if not (folder / "manifest.json").exists():
            raise FileNotFoundError(f"No index in {folder}. Run `rag ingest` first.")
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        with open(folder / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]
        vectors = VectorStore.load(folder / "vectors.npy")
        if len(vectors) != len(chunks):
            raise ValueError("vectors.npy and chunks.jsonl disagree on length; rebuild the index")
        return cls(chunks, vectors, manifest["embed_model"], manifest)
