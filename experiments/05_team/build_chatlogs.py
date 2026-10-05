#!/usr/bin/env python3
"""Export chat logs in the LLMBox session format (see api/llmbox/src/modes.py run_chat).

- chatlogs/dev_train/*.json: one session per DEV training conversation, readable by
  LLMBox mode=finetune with data.type=chat_log.
- chatlogs/finetuned_eval_sessions.jsonl: one session per EVAL item answered by the
  LoRA-finetuned model (T4), with the gold label kept beside the turn for scoring.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / "runs" / "T4_finetuned"
TRAIN = HERE / "finetune" / "train_conversations.jsonl"
OUT = HERE / "chatlogs"
MODEL = "phi-4-mini-instruct"


def session(session_id: str, system_prompt: str, user: str, assistant: str, adapter: str | None) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "session": {
            "session_id": session_id,
            "started_at": now,
            "model": {"name": MODEL, "adapter": adapter},
            "settings": {"system_prompt": system_prompt},
        },
        "turns": [{
            "turn": 1,
            "user": {"username": "resident", "content": user, "timestamp": now},
            "assistant": {"content": assistant, "timestamp": now},
        }],
    }


def export_train() -> int:
    if not TRAIN.exists():
        return 0
    out_dir = OUT / "dev_train"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for line in TRAIN.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        sys_msg, user, asst = (m["content"] for m in r["messages"])
        rec = session(r["doc_id"], sys_msg, user, asst, adapter=None)
        (out_dir / f"{r['doc_id']}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
        n += 1
    return n


def export_eval() -> int:
    resp = RUN / "responses.jsonl"
    if not resp.exists():
        return 0
    from team311.pipeline import SYSTEM, t0_prompt

    rows = [json.loads(l) for l in resp.read_text().splitlines() if l.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "finetuned_eval_sessions.jsonl").open("w") as f:
        for r in rows:
            rec = session(
                r["doc_id"], SYSTEM, t0_prompt(r["input"]),
                r.get("raw") or json.dumps(r.get("pred", {}), ensure_ascii=False),
                adapter="experiments/05_team/finetune/adapter",
            )
            rec["doc_id"] = r["doc_id"]
            rec["subtopic_key"] = r["subtopic_key"]
            rec["gold"] = r["gold"]
            rec["pred"] = r.get("pred")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    import sys

    sys.path.insert(0, str(HERE.parents[1] / "api"))
    print("dev_train sessions:", export_train())
    print("eval sessions:", export_eval())


if __name__ == "__main__":
    main()
