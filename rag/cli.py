"""Command line interface.

    rag ingest data/fastapi              build the index
    rag search "how do I add CORS"       show retrieved chunks, no LLM call
    rag ask "how do I add CORS"          retrieve, then answer with Claude
    rag chat                             ask questions in a loop
    rag eval                             compare retrievers on eval/questions.jsonl
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

DEFAULT_INDEX = "index"


def _retriever(args, need_reranker: bool):
    from .index import Index
    from .models import CrossEncoderReranker, SentenceTransformerEmbedder
    from .retrieval import Retriever

    index = Index.load(args.index)
    embedder = SentenceTransformerEmbedder(index.embed_model)
    reranker = CrossEncoderReranker() if need_reranker else None
    return Retriever(index, embedder, reranker)


def cmd_ingest(args) -> None:
    from .chunking import ChunkerConfig
    from .index import Index
    from .models import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(args.model)
    cfg = ChunkerConfig(max_chars=args.chunk_size, overlap_chars=args.overlap)
    index = Index.build(args.corpus, embedder, cfg)
    index.save(args.index)
    m = index.manifest
    print(f"Indexed {m['documents']} documents as {m['chunks']} chunks in {m['build_seconds']}s -> {args.index}/")


def _print_hits(hits) -> None:
    for h in hits:
        c = h.chunk
        print(f"\n[{h.rank}] {c.source}  ·  {c.section}  (score {h.score:.3f})")
        snippet = " ".join(c.text.split())[:300]
        print(textwrap.indent(textwrap.fill(snippet, 100), "    "))


def cmd_search(args) -> None:
    retriever = _retriever(args, args.rerank)
    _print_hits(retriever.search(args.query, k=args.k, mode=args.mode, rerank=args.rerank))


def _ask_once(retriever, generator, question: str, args) -> None:
    hits = retriever.search(question, k=args.k, mode=args.mode, rerank=args.rerank)
    print()
    answer = generator.answer(question, hits, on_text=lambda t: print(t, end="", flush=True))
    print("\n")
    if answer.refused:
        print("(The model declined to answer this question.)")
    if answer.sources:
        print("Sources:")
        for i, hit in enumerate(answer.sources, start=1):
            mark = "*" if i in answer.cited else " "
            print(f"  {mark}[{i}] {hit.chunk.source}  ·  {hit.chunk.section}")
        print("  (* = cited in the answer)")
    if answer.invalid_citations:
        print(f"  Warning: citations {answer.invalid_citations} do not match any source.")
    if answer.model:
        print(f"  {answer.model} · {answer.input_tokens} in / {answer.output_tokens} out tokens")


def _generator():
    from .generation import Generator

    return Generator()


def cmd_ask(args) -> None:
    _ask_once(_retriever(args, args.rerank), _generator(), args.question, args)


def cmd_chat(args) -> None:
    retriever, generator = _retriever(args, args.rerank), _generator()
    print("Ask about the indexed docs. Empty line or Ctrl+C to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        _ask_once(retriever, generator, question, args)


def cmd_eval(args) -> None:
    from .evaluation import as_markdown, evaluate, load_questions

    questions = load_questions(args.questions)
    retriever = _retriever(args, not args.no_rerank)
    configs = [("bm25", False), ("dense", False), ("hybrid", False)]
    if not args.no_rerank:
        configs.append(("hybrid", True))
    results = []
    for mode, rerank in configs:
        print(f"Evaluating {mode}{' + rerank' if rerank else ''} ...", file=sys.stderr)
        results.append(evaluate(retriever, questions, mode, rerank))
    table = as_markdown(results, len(questions))
    print("\n" + table)
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")
    if args.show_misses:
        for r in results:
            print(f"\nMissed by {r.name} (top 10):")
            for q in r.misses:
                print(f"  - {q}")


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to a legacy code page

    p = argparse.ArgumentParser(prog="rag", description="Retrieval augmented generation, built from scratch.")
    p.add_argument("--index", default=DEFAULT_INDEX, help="index folder (default: index)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="chunk, embed and index a folder of documents")
    s.add_argument("corpus")
    s.add_argument("--model", default=None, help="sentence-transformers model name")
    s.add_argument("--chunk-size", type=int, default=1200)
    s.add_argument("--overlap", type=int, default=200)
    s.set_defaults(func=cmd_ingest)

    def retrieval_flags(sp):
        sp.add_argument("-k", type=int, default=6, help="chunks to retrieve")
        sp.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
        sp.add_argument("--rerank", action="store_true", help="rerank candidates with a cross-encoder")

    s = sub.add_parser("search", help="retrieve chunks without calling an LLM")
    s.add_argument("query")
    retrieval_flags(s)
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("ask", help="answer one question with citations")
    s.add_argument("question")
    retrieval_flags(s)
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("chat", help="ask questions in a loop")
    retrieval_flags(s)
    s.set_defaults(func=cmd_chat)

    s = sub.add_parser("eval", help="measure retrieval quality")
    s.add_argument("--questions", default="eval/questions.jsonl")
    s.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder run")
    s.add_argument("--out", help="also write the results table to this file")
    s.add_argument("--show-misses", action="store_true")
    s.set_defaults(func=cmd_eval)

    args = p.parse_args(argv)
    if args.cmd == "ingest" and args.model is None:
        from .models import DEFAULT_EMBED_MODEL

        args.model = DEFAULT_EMBED_MODEL
    args.func(args)


if __name__ == "__main__":
    main()
