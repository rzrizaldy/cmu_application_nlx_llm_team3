"""Knowledge index for retrieval (no labeled intake examples)."""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "corpora" / "05_all" / "corpus.jsonl"
OPS = REPO / "corpora" / "05_all" / "operational_evidence.jsonl"

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


def load_knowledge(corpus_path: Path = CORPUS, ops_path: Path = OPS) -> list[dict]:
    rows = [json.loads(line) for line in corpus_path.read_text().splitlines() if line.strip()]
    if ops_path.exists():
        rows += [json.loads(line) for line in ops_path.read_text().splitlines() if line.strip()]
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
            "stats": r["table_json"] if r["metadata"]["member"] == "team" else None,
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


def _fmt_days(d: float) -> str:
    return f"{d * 24:.0f} hours" if d < 1 else f"{d:.1f} days"


def resolution_range(index: list[dict], issue: str | None = None, category: str | None = None) -> str | None:
    """Historical closure time from the WPRDC operational evidence.

    Exact issue match first; otherwise the volume-weighted median across the
    category's issues, with the 75th-90th percentile span of its busiest issue.
    """
    stats = [d["stats"] for d in index if d["stats"] and "median_days" in d["stats"]]
    if issue:
        hit = next((s for s in stats if s["issue"].strip().lower() == issue.strip().lower()), None)
        if hit:
            return (f"{hit['issue']}: median {_fmt_days(hit['median_days'])}, "
                    f"75th-90th percentile {_fmt_days(hit['p75_days'])}-{_fmt_days(hit['p90_days'])} "
                    f"({hit['closed_requests']} closed requests)")
    if category:
        rows = sorted((s for s in stats if s["category"] == category), key=lambda s: -s["closed_requests"])
        if rows:
            total = sum(s["closed_requests"] for s in rows)
            median = sum(s["median_days"] * s["closed_requests"] for s in rows) / max(total, 1)
            top = rows[0]
            return (f"{category}: median about {_fmt_days(median)} across {len(rows)} issues "
                    f"({total} closed requests); busiest issue {top['issue']} "
                    f"75th-90th percentile {_fmt_days(top['p75_days'])}-{_fmt_days(top['p90_days'])}")
    return None
