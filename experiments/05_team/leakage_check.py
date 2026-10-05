#!/usr/bin/env python3
"""Fail if any EVAL item can reach the model through retrieval or DEV neighbors.

Checks doc_id overlap and exact input-text overlap against the knowledge index
and against dev.jsonl (the labeled neighbors used for issue voting and finetuning).
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "api"))
from team311.knowledge import load_knowledge  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"


def rows(name: str) -> list[dict]:
    return [json.loads(l) for l in (DATA / name).read_text().splitlines() if l.strip()]


def key(text: str) -> str:
    return " ".join(text.lower().split())


ev, dev, kb = rows("eval.jsonl"), rows("dev.jsonl"), load_knowledge()
eval_ids, eval_texts = {r["doc_id"] for r in ev}, {key(r["input"]) for r in ev}
problems = {
    "eval ids in knowledge index": eval_ids & {d["doc_id"] for d in kb},
    "eval ids in dev": eval_ids & {r["doc_id"] for r in dev},
    "eval texts in dev": eval_texts & {key(r["input"]) for r in dev},
    "eval texts in knowledge index": {t for t in eval_texts if len(t) > 40} & {key(d["text"]) for d in kb},
}
failed = {k: sorted(v)[:5] for k, v in problems.items() if v}
if failed:
    print("FAIL:", json.dumps(failed, indent=2))
    sys.exit(1)
print(f"OK: {len(ev)} eval items; none in the {len(kb)}-doc knowledge index or the {len(dev)} dev neighbors")
