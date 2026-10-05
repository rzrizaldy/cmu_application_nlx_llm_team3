"""Knowledge index for retrieval (no labeled intake examples)."""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "corpora" / "05_all" / "corpus.jsonl"

INTAKE_MARKERS = {
    "afaq": lambda m: m.get("document_type") == "complaint",
    "mahika": lambda m: m.get("document_type") == "service_request",
    "mingchin": lambda m: bool(m.get("example_style")),
    "rutomo": lambda m: m.get("source_kind") == "issue_taxonomy",
}


def _content(record: dict) -> str:
    text = record.get("raw_text") or ""
    table = record.get("table_json")
    if table is not None:
        text += "\n" + json.dumps(table, ensure_ascii=False)
    return text.strip()


def load_knowledge(corpus_path: Path = CORPUS) -> list[dict]:
    rows = [json.loads(line) for line in corpus_path.read_text().splitlines() if line.strip()]
    out = []
    for r in rows:
        member = r["metadata"]["member"]
        if INTAKE_MARKERS.get(member, lambda _: False)(r["metadata"]):
            continue
        out.append({
            "doc_id": r["doc_id"],
            "subtopic_key": r["metadata"]["subtopic_key"],
            "text": _content(r),
            "kind": r["metadata"].get("document_type")
            or r["metadata"].get("source_kind")
            or r["metadata"].get("record_type")
            or "record",
        })
    return out


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def retrieve(query: str, index: list[dict], subtopic_key: str | None = None, k: int = 3) -> list[dict]:
    q = tokenize(query)
    scored = []
    for doc in index:
        if subtopic_key and doc["subtopic_key"] != subtopic_key:
            continue
        t = tokenize(doc["text"])
        if not t:
            continue
        scored.append((len(q & t) / max(len(q), 1), doc))
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored[:k]]


def resolution_range_for_category(category: str, index: list[dict]) -> str | None:
    for doc in index:
        if doc["kind"] != "operational_summary" and doc["kind"] != "resolution_statistics":
            continue
        if category.lower() in doc["text"].lower():
            return doc["text"][:400]
    return None
