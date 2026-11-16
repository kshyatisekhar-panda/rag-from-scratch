"""Loading source files into Document objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst"}


@dataclass
class Document:
    source: str  # path relative to the corpus root, with forward slashes
    text: str
    title: str = ""


@dataclass
class Chunk:
    id: int
    source: str
    section: str  # heading trail, for example "Path Parameters > Data conversion"
    text: str
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "source": self.source, "section": self.section, "text": self.text, "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(id=d["id"], source=d["source"], section=d["section"], text=d["text"], meta=d.get("meta", {}))


def _title_of(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def load_directory(root: str | Path) -> list[Document]:
    """Read every text file under root, sorted so builds are reproducible."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus folder not found: {root}")
    docs = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            continue
        rel = path.relative_to(root).as_posix()
        docs.append(Document(source=rel, text=text, title=_title_of(text, path.stem)))
    return docs
