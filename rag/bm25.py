"""Okapi BM25 over an inverted index, written out in full.

score(q, d) = sum over terms t in q of
    idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))

idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))   (the Lucene variant, never negative)

Keyword search catches what embeddings miss: exact identifiers such as
`HTTPException`, `Depends` or `status_code=201`.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

TOKEN = re.compile(r"[a-z0-9_]+")

# Small on purpose. Words like "not", "how" and "with" carry meaning in technical questions.
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will".split()
)


def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in TOKEN.findall(text.lower()):
        if tok in STOPWORDS:
            continue
        tokens.append(tok)
        # Also index the parts of snake_case identifiers: "response_model" matches "response model".
        if "_" in tok:
            tokens.extend(p for p in tok.split("_") if p and p not in STOPWORDS)
    return tokens


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)  # term -> [(doc, tf)]
        self.doc_len: list[int] = []
        self.avgdl = 0.0
        self.idf: dict[str, float] = {}

    def fit(self, texts: list[str]) -> "BM25":
        self.postings.clear()
        self.doc_len = []
        for doc_id, text in enumerate(texts):
            counts = Counter(tokenize(text))
            self.doc_len.append(sum(counts.values()))
            for term, tf in counts.items():
                self.postings[term].append((doc_id, tf))
        n = len(texts)
        self.avgdl = (sum(self.doc_len) / n) if n else 0.0
        self.idf = {
            term: math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5)) for term, plist in self.postings.items()
        }
        return self

    def scores(self, query: str) -> dict[int, float]:
        """Sparse scores: only documents that share at least one term with the query."""
        out: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf[term]
            for doc_id, tf in plist:
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[doc_id] / self.avgdl)
                out[doc_id] += idf * tf * (self.k1 + 1) / (tf + norm)
        return out

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        ranked = sorted(self.scores(query).items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]
