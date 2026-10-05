"""
score_adjudication.py -- Part D human-adjudication scorer.

After you hand-label out/partD_adjudication_TEMPLATE.jsonl (set
guard_decision_correct to true/false for each row), save it as
out/partD_adjudication.jsonl and run this. It reports agreement between your
labels and the guardrail's decisions and lists the disagreements -- the same
human-vs-instrument check as Assignment 1 Sec 3.1, applied to the guardrail.

    python score_adjudication.py
"""
from __future__ import annotations
import json
from pathlib import Path

import scenario


def main():
    p = scenario.OUT_DIR / "partD_adjudication.jsonl"
    if not p.exists():
        raise SystemExit(
            f"{p} not found. Copy partD_adjudication_TEMPLATE.jsonl to "
            "partD_adjudication.jsonl and fill guard_decision_correct (true/false).")
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    labeled = [r for r in rows if str(r.get("guard_decision_correct")).lower() in ("true", "false")]
    if not labeled:
        raise SystemExit("No rows labeled yet. Set guard_decision_correct on each row.")
    agree = sum(1 for r in labeled if str(r["guard_decision_correct"]).lower() == "true")
    n = len(labeled)
    disagreements = [r for r in labeled if str(r["guard_decision_correct"]).lower() == "false"]
    report = {
        "n_adjudicated": n,
        "guard_agreement_rate": round(agree / n, 3),
        "n_disagreements": len(disagreements),
        "disagreements": [{"probe_id": r["probe_id"], "category": r.get("category"),
                           "guard_allowed": r.get("guard_allowed")} for r in disagreements],
    }
    (scenario.OUT_DIR / "partD_adjudication_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
