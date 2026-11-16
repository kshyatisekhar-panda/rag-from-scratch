"""Markdown aware chunking.

The strategy, in order:
1. Split a document into sections at headings, remembering the heading trail
   ("Security > OAuth2 with JWT > Hash passwords"). Fenced code is tracked so a
   "#" comment inside a code block is never mistaken for a heading.
2. Split each section into blocks: paragraphs, lists, and whole code fences.
3. Greedily pack blocks into chunks up to max_chars. The last block of a chunk is
   repeated at the start of the next one when it is short enough, which gives
   overlap on natural boundaries instead of cutting words in half.
4. A block that is larger than max_chars on its own is split by lines, then by
   sentences, then by hard character windows as a last resort.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .documents import Chunk, Document

HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ChunkerConfig:
    max_chars: int = 1200
    overlap_chars: int = 200
    min_chars: int = 40


def split_sections(text: str) -> list[tuple[list[str], str]]:
    """Return (heading_trail, body) pairs in document order."""
    sections: list[tuple[list[str], str]] = []
    trail: list[tuple[int, str]] = []
    body: list[str] = []
    in_fence = False

    def flush():
        content = "\n".join(body).strip()
        if content:
            sections.append(([t for _, t in trail], content))
        body.clear()

    for line in text.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            body.append(line)
            continue
        m = None if in_fence else HEADING.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), _clean_heading(m.group(2))
            while trail and trail[-1][0] >= level:
                trail.pop()
            trail.append((level, title))
        else:
            body.append(line)
    flush()
    return sections


def _clean_heading(title: str) -> str:
    # MkDocs anchors like "Query Parameters { #query-parameters }"
    return re.sub(r"\s*\{\s*#[^}]*\}\s*$", "", title).strip()


def split_blocks(body: str) -> list[str]:
    """Paragraphs separated by blank lines, keeping each code fence whole."""
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in body.splitlines():
        if FENCE.match(line):
            if not in_fence and current:
                blocks.append("\n".join(current))
                current = []
            current.append(line)
            in_fence = not in_fence
            if not in_fence:
                blocks.append("\n".join(current))
                current = []
            continue
        if not in_fence and not line.strip():
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current))
    return [b.strip("\n") for b in blocks if b.strip()]


def split_oversized(block: str, max_chars: int) -> list[str]:
    """Break one block that does not fit, preferring line then sentence boundaries."""
    if len(block) <= max_chars:
        return [block]
    units = block.splitlines() if "\n" in block else SENTENCE_END.split(block)
    joiner = "\n" if "\n" in block else " "
    pieces: list[str] = []
    current = ""
    for unit in units:
        if len(unit) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(unit[i : i + max_chars] for i in range(0, len(unit), max_chars))
            continue
        candidate = f"{current}{joiner}{unit}" if current else unit
        if len(candidate) > max_chars:
            pieces.append(current)
            current = unit
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def pack(blocks: list[str], cfg: ChunkerConfig) -> list[str]:
    units = [piece for b in blocks for piece in split_oversized(b, cfg.max_chars)]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for unit in units:
        added = len(unit) + (2 if current else 0)
        if current and size + added > cfg.max_chars:
            chunks.append("\n\n".join(current))
            tail = current[-1]
            if len(tail) <= cfg.overlap_chars and len(tail) + 2 + len(unit) <= cfg.max_chars:
                current, size = [tail], len(tail)
            else:
                current, size = [], 0
            added = len(unit) + (2 if current else 0)
        current.append(unit)
        size += added
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_document(doc: Document, cfg: ChunkerConfig, start_id: int = 0) -> list[Chunk]:
    chunks: list[Chunk] = []
    for trail, body in split_sections(doc.text):
        section = " > ".join(trail) if trail else doc.title
        for text in pack(split_blocks(body), cfg):
            if len(text.strip()) < cfg.min_chars:
                continue
            chunks.append(Chunk(id=start_id + len(chunks), source=doc.source, section=section, text=text))
    return chunks


def chunk_documents(docs: list[Document], cfg: ChunkerConfig | None = None) -> list[Chunk]:
    cfg = cfg or ChunkerConfig()
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, cfg, start_id=len(out)))
    return out


def searchable_text(chunk: Chunk) -> str:
    """What gets embedded and keyword indexed: the heading trail adds context a bare paragraph lacks."""
    return f"{chunk.section}\n\n{chunk.text}"
