"""Repository guard: fails if the tree contains things that must never be published.

- Windows absolute paths (personal folders such as a drive letter followed by a backslash)
- model weights or audio files
- secret-looking strings (API tokens, passwords)

Run: python tools/guard.py   (exit code 1 when something is found)
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", "venv", ".venv", "cache", "output", "tmp", "__pycache__", ".pytest_cache", "node_modules"}
SELF = Path(__file__).resolve()
BLOCKED_EXT = {".onnx", ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".wav", ".mp3"}
CHECKS = [
    ("Windows absolute path", re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:\\")),
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{20,}")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("password assignment", re.compile(r"password\s*=", re.IGNORECASE)),
]


def main() -> int:
    problems = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            p = Path(dirpath) / name
            rel = p.relative_to(ROOT).as_posix()
            if p.suffix.lower() == ".mp3" and rel.startswith("docs/"):
                pass  # demo samples for the GitHub Pages site, same rule as .gitignore
            elif p.suffix.lower() in BLOCKED_EXT:
                problems.append(f"{rel}: weight/audio file must not be committed")
                continue
            if p.resolve() == SELF:
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                for label, rx in CHECKS:
                    if rx.search(line):
                        problems.append(f"{rel}:{lineno}: {label}: {line.strip()[:120]}")
    for msg in problems:
        print(msg)
    print(f"guard: {len(problems)} problem(s) found" if problems else "guard: OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
