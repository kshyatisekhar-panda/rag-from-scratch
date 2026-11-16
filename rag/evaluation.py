"""Retrieval evaluation.

Each line of the question file is {"question": ..., "relevant": ["path/to/file.md", ...]}.
A hit counts when a retrieved chunk comes from one of the relevant files.

Metrics
    hit@k   share of questions with at least one relevant chunk in the top k
    MRR     mean of 1 / rank of the first relevant chunk (0 when none in the top 10)
    ms      mean retrieval latency per query

This measures retrieval only and spends no API credits. Retrieval is where most
RAG failures start: the model cannot cite a passage it was never shown.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from .retrieval import Retriever


@dataclass
class EvalResult:
    name: str
    hit_at: dict[int, float]
    mrr: float
    ms: float
    misses: list[str]


def load_questions(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(retriever: Retriever, questions: list[dict], mode: str, rerank: bool, ks=(1, 3, 5, 10)) -> EvalResult:
    depth = max(ks)
    hits_at = {k: 0 for k in ks}
    rr_total = 0.0
    misses = []
    elapsed = 0.0
    for q in questions:
        relevant = set(q["relevant"])
        t0 = time.perf_counter()
        results = retriever.search(q["question"], k=depth, mode=mode, rerank=rerank)
        elapsed += time.perf_counter() - t0
        first = next((h.rank for h in results if h.chunk.source in relevant), None)
        if first is None:
            misses.append(q["question"])
            continue
        rr_total += 1.0 / first
        for k in ks:
            if first <= k:
                hits_at[k] += 1
    n = len(questions)
    name = mode + (" + rerank" if rerank else "")
    return EvalResult(name, {k: v / n for k, v in hits_at.items()}, rr_total / n, 1000 * elapsed / n, misses)


def as_markdown(results: list[EvalResult], n_questions: int) -> str:
    ks = sorted(results[0].hit_at)
    head = "| Retriever | " + " | ".join(f"hit@{k}" for k in ks) + " | MRR | ms/query |"
    sep = "|---|" + "---|" * (len(ks) + 2)
    rows = [
        f"| {r.name} | " + " | ".join(f"{r.hit_at[k]:.2f}" for k in ks) + f" | {r.mrr:.3f} | {r.ms:.0f} |"
        for r in results
    ]
    return "\n".join([f"{n_questions} questions", "", head, sep, *rows])
