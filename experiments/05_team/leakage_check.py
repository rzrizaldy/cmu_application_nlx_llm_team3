#!/usr/bin/env python3
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "api"))
from team311.knowledge import load_knowledge  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
eval_ids = {json.loads(l)["doc_id"] for l in (DATA / "eval.jsonl").read_text().splitlines() if l.strip()}
kb_ids = {d["doc_id"] for d in load_knowledge()}
overlap = eval_ids & kb_ids
if overlap:
    print("FAIL: eval doc_ids in knowledge index:", overlap)
    sys.exit(1)
print("OK: no eval leakage into knowledge index")
