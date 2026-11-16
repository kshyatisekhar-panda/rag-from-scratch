# Changelog

## 0.1.0 · 2026-11-16

First release.

### Added
- Markdown chunker that splits at headings, keeps code blocks whole and records the heading trail of every chunk.
- Okapi BM25 keyword search on an inverted index, with snake_case identifiers also indexed by their parts.
- Exact cosine vector search in numpy, with local embeddings from `BAAI/bge-small-en-v1.5`.
- Hybrid retrieval with Reciprocal Rank Fusion and optional cross-encoder reranking.
- Answers from Claude that cite numbered sources, with every citation checked in code.
- Retrieval eval (hit@k, MRR, latency) on 38 questions over the FastAPI docs.
- `rag` command line tool: `ingest`, `search`, `ask`, `chat`, `eval`.
- Script that downloads the FastAPI docs and inlines their code examples.
- Unit tests and a CI workflow for Python 3.10 to 3.12.
