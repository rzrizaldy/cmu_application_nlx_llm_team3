"""Tool helpers for codebook lookup."""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CODEBOOK = REPO / "corpora" / "04_parks_public_spaces_mingchin" / "311_issue_category_codebook.csv"


def _rows() -> list[dict]:
    if not CODEBOOK.exists():
        return []
    with CODEBOOK.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def lookup_codebook(text: str, categories: list[str] | None = None, k: int = 5) -> list[dict]:
    """Codebook rows whose issue name appears in the text, longest names first."""
    text_l = text.lower()
    seen, hits = set(), []
    for row in sorted(_rows(), key=lambda r: -len(r.get("issue") or "")):
        issue = (row.get("issue") or "").strip()
        if len(issue) < 4 or issue.lower() not in text_l:
            continue
        if categories and row.get("category") not in categories:
            continue
        key = (issue, row.get("category"), row.get("department"))
        if key in seen:
            continue
        seen.add(key)
        hits.append({
            "request_type_id": row.get("request_type_id"),
            "issue": issue,
            "category": row.get("category"),
            "department": row.get("department"),
        })
        if len(hits) >= k:
            break
    return hits


def lookup_codebook_json(text: str, categories: list[str] | None = None) -> str:
    return json.dumps(lookup_codebook(text, categories), ensure_ascii=False)
