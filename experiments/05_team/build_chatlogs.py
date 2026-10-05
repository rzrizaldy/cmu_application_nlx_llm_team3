#!/usr/bin/env python3
"""Export T4 eval sessions as LLMBox-style chat log JSON for appendix."""
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

RUN = Path(__file__).resolve().parent / "runs" / "T4_finetuned"
OUT = Path(__file__).resolve().parent / "chatlogs"


def main() -> None:
    resp = RUN / "responses.jsonl"
    if not resp.exists():
        print("Missing", resp)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in resp.read_text().splitlines() if l.strip()]
    export = []
    for r in rows:
        session_id = uuid4().hex[:8]
        export.append({
            "session_id": session_id,
            "doc_id": r["doc_id"],
            "turns": [
                {"role": "user", "content": r["input"]},
                {"role": "assistant", "content": json.dumps(r.get("pred", {}), ensure_ascii=False)},
            ],
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        })
    out_path = OUT / "finetuned_eval_sessions.jsonl"
    out_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in export))
    print("wrote", out_path, len(export))


if __name__ == "__main__":
    main()
