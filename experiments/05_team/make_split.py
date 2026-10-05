#!/usr/bin/env python3
"""Build stratified DEV/EVAL intake splits from corpora/05_all.

Gold labels come from the WPRDC codebook (via operational_evidence.jsonl), as
the brief's label contract requires: request_type_id, issue, exact category,
and department. Member labels are kept under gold.member_label; items whose
issue cannot be resolved to a codebook request type keep the member label and
are marked gold_source=member.
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "corpora" / "05_all" / "corpus.jsonl"
OPS = REPO / "corpora" / "05_all" / "operational_evidence.jsonl"
OUT = Path(__file__).resolve().parent / "data"
SEED = 952
EVAL_N = 50

sys.path.insert(0, str(REPO / "corpora" / "05_all"))
from category_groups import CATEGORY_GROUPS  # noqa: E402

CATEGORY_TO_DOMAIN = {c: sk for sk, cats in CATEGORY_GROUPS.items() for c in cats}
ROUTED_SENTENCE = re.compile(r"\s*It was routed to [^.]*\.")
STATUS_SENTENCE = re.compile(r"\s*Current status: [^.]*\.")


def issue_key(name: str | None) -> str:
    return re.sub(r"\s*\(do not use\)\s*$", "", (name or "").strip().lower())


def load_codebook_index() -> tuple[dict, dict]:
    by_id, by_name = {}, {}
    for line in OPS.read_text().splitlines():
        t = json.loads(line)["table_json"]
        label = {k: t[k] for k in ("request_type_id", "issue", "category", "department")}
        by_id[t["request_type_id"]] = label
        for name in [t["issue"], *t.get("aliases", [])]:
            by_name.setdefault(issue_key(name), label)
    return by_id, by_name


def is_intake(record: dict) -> bool:
    m = record["metadata"]
    if m["member"] == "afaq" and m.get("document_type") == "complaint":
        return True
    if m["member"] == "mahika":
        return True
    if m["member"] == "mingchin" and m.get("example_style"):
        return True
    if m["member"] == "rutomo" and m.get("source_kind") == "issue_taxonomy":
        return True
    return False


def _table(record: dict) -> dict:
    t = record.get("table_json") or {}
    if isinstance(t, list):
        t = t[0] if t else {}
    return t


def member_label(record: dict) -> dict:
    m, t = record["metadata"], _table(record)
    return {
        "request_type_id": str(m.get("request_type_id") or t.get("request_type_id") or "") or None,
        "issue": m.get("issue") or m.get("request_type") or t.get("issue") or t.get("request_type") or "",
        "category": m.get("intended_category") or m.get("codebook_category") or t.get("category") or m.get("category"),
        "department": m.get("intended_department") or t.get("department") or m.get("department"),
    }


def gold(record: dict, by_id: dict, by_name: dict) -> dict:
    m = record["metadata"]
    mine = member_label(record)
    cb = by_id.get(mine["request_type_id"] or "") or by_name.get(issue_key(mine["issue"]))
    origin = "complaint" if m.get("document_type") == "complaint" else (
        "codebook_row" if m.get("source_kind") == "issue_taxonomy" else "intake"
    )
    base = cb or mine
    domain = CATEGORY_TO_DOMAIN.get(base["category"] or "", m["subtopic_key"])
    return {**base, "domain": domain, "input_origin": origin, "gold_source": "codebook" if cb else "member", "member_label": mine}


def intake_text(record: dict) -> str:
    """Text the model sees. Strips fields that state the answer outright."""
    m = record["metadata"]
    if m.get("source_kind") == "issue_taxonomy":
        t = _table(record)
        text = f"Resident reports: {t.get('issue', '')}."
        if t.get("definition"):
            text += f" {t['definition']}"
        return text
    text = (record.get("raw_text") or "").strip()
    if m["member"] == "mahika":
        text = STATUS_SENTENCE.sub("", ROUTED_SENTENCE.sub("", text)).strip()
    return text


def row(record: dict, by_id: dict, by_name: dict) -> dict:
    return {
        "doc_id": record["doc_id"],
        "subtopic_key": record["metadata"]["subtopic_key"],
        "member": record["metadata"]["member"],
        "input": intake_text(record),
        "gold": gold(record, by_id, by_name),
    }


def main() -> None:
    records = [json.loads(line) for line in CORPUS.read_text().splitlines() if line.strip()]
    by_id, by_name = load_codebook_index()
    pool = [row(r, by_id, by_name) for r in records if is_intake(r)]
    by_key: dict[str, list] = defaultdict(list)
    for item in pool:
        by_key[item["subtopic_key"]].append(item)

    rng = random.Random(SEED)
    eval_ids: set[str] = set()
    eval_rows: list[dict] = []
    per = max(1, EVAL_N // len(by_key))
    for key, items in sorted(by_key.items()):
        rng.shuffle(items)
        take = items[:per]
        eval_rows.extend(take)
        eval_ids.update(x["doc_id"] for x in take)
    if len(eval_rows) < EVAL_N:
        rest = [x for x in pool if x["doc_id"] not in eval_ids]
        rng.shuffle(rest)
        for x in rest:
            if len(eval_rows) >= EVAL_N:
                break
            eval_rows.append(x)
            eval_ids.add(x["doc_id"])
    eval_rows = eval_rows[:EVAL_N]
    eval_ids = {x["doc_id"] for x in eval_rows}
    dev_rows = [x for x in pool if x["doc_id"] not in eval_ids]

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": SEED,
        "eval_n": len(eval_rows),
        "dev_n": len(dev_rows),
        "by_subtopic_eval": {k: sum(1 for r in eval_rows if r["subtopic_key"] == k) for k in by_key},
        "gold_source": {
            split: dict(Counter(f"{r['subtopic_key']}:{r['gold']['gold_source']}" for r in rows))
            for split, rows in (("eval", eval_rows), ("dev", dev_rows))
        },
        "member_vs_codebook_disagreement": {
            field: sum(
                1 for r in pool
                if r["gold"]["gold_source"] == "codebook"
                and (r["gold"]["member_label"][field] or "") != (r["gold"][field] or "")
            )
            for field in ("category", "department")
        },
        "gold_domain_differs_from_member_corpus": sum(1 for r in pool if r["gold"]["domain"] != r["subtopic_key"]),
    }
    (OUT / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (OUT / "dev.jsonl").open("w") as f:
        for r in dev_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (OUT / "eval.jsonl").open("w") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
