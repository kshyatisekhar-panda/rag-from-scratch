"""Download the FastAPI documentation (MIT licensed) as a demo corpus.

Uses a shallow sparse git clone so only docs/en/docs and docs_src are fetched.
The docs pull their code samples in with include directives such as
    {* ../../docs_src/path_params/tutorial001.py hl[6:7] *}
Those are replaced with the real code in a fenced block, because the examples
are exactly what developers search for.

    python scripts/fetch_fastapi_docs.py            # writes data/fastapi/
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

REPO = "https://github.com/fastapi/fastapi.git"

# Pages that are mostly lists of names, changelogs or API stubs generated at build time.
SKIP_PREFIXES = ("reference/",)
SKIP_FILES = {
    "release-notes.md", "fastapi-people.md", "external-links.md", "contributing.md", "management.md",
    "management-tasks.md", "translations.md", "newsletter.md", "help-fastapi.md", "_llm-test.md",
    "agent-banner.md", "translation-banner.md",
}

INCLUDE_STAR = re.compile(r"\{\*\s*(\S+)[^*]*\*\}")  # {* path hl[..] *}
INCLUDE_BANG = re.compile(r"\{!>?\s*(\S+?)\s*!\}")  # {! path !}


def run(*cmd: str, cwd: Path | None = None) -> str:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def clone(cache: Path) -> str:
    if not (cache / ".git").exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, str(cache))
        run("git", "sparse-checkout", "set", "docs/en/docs", "docs_src", cwd=cache)
    return run("git", "rev-parse", "HEAD", cwd=cache)


def resolve_includes(text: str, docs_src: Path) -> tuple[str, int]:
    count = 0

    def replace(match: re.Match) -> str:
        nonlocal count
        ref = match.group(1)
        if "docs_src/" not in ref:
            return match.group(0)
        path = docs_src / ref.split("docs_src/", 1)[1]
        if not path.is_file():
            return ""
        count += 1
        lang = {".py": "python", ".toml": "toml", ".sh": "bash", ".json": "json"}.get(path.suffix, "")
        return f"```{lang}\n{path.read_text(encoding='utf-8').rstrip()}\n```"

    text = INCLUDE_STAR.sub(replace, text)
    text = INCLUDE_BANG.sub(replace, text)
    return text, count


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/fastapi")
    p.add_argument("--cache", default="data/.cache/fastapi")
    args = p.parse_args()

    cache, out = Path(args.cache), Path(args.out)
    commit = clone(cache)
    docs, docs_src = cache / "docs/en/docs", cache / "docs_src"

    if out.exists():
        shutil.rmtree(out)
    written = includes = 0
    for src in sorted(docs.rglob("*.md")):
        rel = src.relative_to(docs).as_posix()
        if rel in SKIP_FILES or rel.startswith(SKIP_PREFIXES):
            continue
        text, n = resolve_includes(src.read_text(encoding="utf-8"), docs_src)
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        written += 1
        includes += n

    (out / "_source.json").write_text(
        json.dumps({"repo": REPO, "commit": commit, "license": "MIT", "files": written}, indent=2), encoding="utf-8"
    )
    print(f"Wrote {written} pages with {includes} code examples inlined to {out}/ (fastapi@{commit[:7]})")


if __name__ == "__main__":
    main()
