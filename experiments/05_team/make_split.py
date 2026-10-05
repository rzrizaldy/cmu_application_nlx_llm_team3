#!/usr/bin/env python3
"""Build stratified DEV/EVAL intake splits from corpora/05_all."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "corpora" / "05_all" / "corpus.jsonl"
OUT = Path(__file__).resolve().parent / "data"
SEED = 952
EVAL_N = 50


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


def gold(record: dict) -> dict:
    m, t = record["metadata"], record.get("table_json") or {}
    if isinstance(t, list):
        t = t[0] if t else {}
    cat = (
        m.get("intended_category")
        or m.get("codebook_category")
        or t.get("category")
        or m.get("category")
    )
    dep = m.get("intended_department") or t.get("department") or m.get("department")
    issue = m.get("request_type") or t.get("issue") or ""
    origin = "complaint" if m.get("document_type") == "complaint" else (
        "codebook_row" if m.get("source_kind") == "issue_taxonomy" else "intake"
    )
    return {"category": cat, "department": dep, "issue": issue, "input_origin": origin}


def row(record: dict) -> dict:
    text = record.get("raw_text") or ""
    if not text.strip() and record.get("table_json"):
        text = json.dumps(record["table_json"], ensure_ascii=False)
    g = gold(record)
    return {
        "doc_id": record["doc_id"],
        "subtopic_key": record["metadata"]["subtopic_key"],
        "member": record["metadata"]["member"],
        "input": text.strip(),
        "gold": g,
    }


def main() -> None:
    records = [json.loads(line) for line in CORPUS.read_text().splitlines() if line.strip()]
    pool = [row(r) for r in records if is_intake(r)]
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
