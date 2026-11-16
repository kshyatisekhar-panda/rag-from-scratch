"""Grounded answer generation with Claude.

The retrieved chunks are numbered and placed in the prompt. Claude is told to
answer only from them and to cite them as [1], [2]. After the answer streams
back, the citations are checked in code: every number must point at a source
that was actually given, so a made up reference is caught instead of shown.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable

import anthropic

from .retrieval import Hit

MODEL = os.environ.get("RAG_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("RAG_EFFORT", "low")  # grounded Q&A rarely needs deep reasoning
MAX_CONTEXT_CHARS = 24_000

SYSTEM_PROMPT = """You answer developer questions using only the documentation excerpts provided in <sources>.

How to answer:
- Base every claim on the sources. Cite them inline with their number in square brackets, like [2] or [1][3], right after the sentence they support.
- Include a short code example when the sources contain one that fits the question.
- If the sources do not contain the answer, say so plainly and mention what they do cover. Do not fill gaps from general knowledge.
- Be direct. No preamble, no closing summary."""

CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    text: str
    sources: list[Hit]
    cited: list[int] = field(default_factory=list)  # 1 based source numbers Claude used
    invalid_citations: list[int] = field(default_factory=list)  # numbers that point at nothing
    refused: bool = False
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


def build_context(hits: list[Hit], max_chars: int = MAX_CONTEXT_CHARS) -> tuple[str, list[Hit]]:
    """Number the sources and stop before the character budget is exceeded."""
    parts, used, total = [], [], 0
    for hit in hits:
        c = hit.chunk
        section = c.section.replace('"', "'")
        block = f'<source id="{len(used) + 1}" file="{c.source}" section="{section}">\n{c.text}\n</source>'
        if used and total + len(block) > max_chars:
            break
        parts.append(block)
        used.append(hit)
        total += len(block)
    return "<sources>\n" + "\n\n".join(parts) + "\n</sources>", used


def check_citations(text: str, n_sources: int) -> tuple[list[int], list[int]]:
    found = sorted({int(m) for m in CITATION.findall(text)})
    valid = [n for n in found if 1 <= n <= n_sources]
    invalid = [n for n in found if not 1 <= n <= n_sources]
    return valid, invalid


class Generator:
    def __init__(self, client: anthropic.Anthropic | None = None, model: str = MODEL, effort: str = EFFORT):
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.effort = effort

    def answer(self, question: str, hits: list[Hit], on_text: Callable[[str], None] | None = None) -> Answer:
        context, used = build_context(hits)
        if not used:
            return Answer(text="No relevant documentation was found for this question.", sources=[])

        with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            output_config={"effort": self.effort},
            # If a safety classifier declines, the API retries on a suitable model inside the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": f"{context}\n\n<question>{question}</question>"}],
        ) as stream:
            for text in stream.text_stream:
                if on_text:
                    on_text(text)
            final = stream.get_final_message()

        text = "".join(b.text for b in final.content if b.type == "text")
        refused = final.stop_reason == "refusal"
        cited, invalid = check_citations(text, len(used))
        return Answer(
            text=text,
            sources=used,
            cited=cited,
            invalid_citations=invalid,
            refused=refused,
            model=final.model,
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
        )
