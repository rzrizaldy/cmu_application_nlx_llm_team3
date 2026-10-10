#!/usr/bin/env python3
"""Build team3_final_submission.zip for Canvas (light: no venv, weights, or secrets)."""
from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "appendix"
OUT = REPO / "team3_final_submission.zip"

SKIP_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "adapter",
    "checkpoints",
    "models",
    "outputs",
    "node_modules",
}
SKIP_SUFFIX = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}
SKIP_NAMES = {".env", "311_data.csv"}


def tracked_files() -> list[str]:
    raw = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO,
        capture_output=True,
        check=True,
    ).stdout
    out = raw.decode().split("\0")
    return sorted(filter(None, out))


def include_path(rel: str) -> bool:
    p = Path(rel)
    if p.name in SKIP_NAMES:
        return False
    if p.suffix.lower() in SKIP_SUFFIX:
        return False
    if any(part in SKIP_PARTS for part in p.parts):
        return False
    if rel.startswith("appendix/"):
        return True
    if rel.startswith("corpora/") and rel.endswith(".jsonl"):
        return True
    if rel.startswith("experiments/05_team/data/") and rel.endswith(".jsonl"):
        return True
    return False


def extra_paths() -> list[str]:
    """Paths required for Canvas but not yet in git."""
    out: list[str] = []
    for p in sorted((APP / "ai_disclosure").glob("*.txt")):
        out.append(str(p.relative_to(REPO)))
    return out


def main() -> None:
    from build_appendix import main as build_appendix  # noqa: WPS433

    build_appendix()
    rels = set(tracked_files()) | set(extra_paths())
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in sorted(rels):
            if not include_path(rel):
                continue
            path = REPO / rel
            if not path.is_file():
                continue
            zf.write(path, f"team3_final_submission/{rel}")
    mb = OUT.stat().st_size / (1024 * 1024)
    print(f"wrote {OUT} ({mb:.2f} MB)")


if __name__ == "__main__":
    main()
