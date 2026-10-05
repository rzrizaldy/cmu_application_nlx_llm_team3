"""
run_partC.py -- Part C: the custom LLM API, four evaluations over 50 held-out
eval inputs.

  Custom Eval 1: structured_output mode with the Pydantic TriageResult schema.
  Custom Eval 2: tool_calling mode -- model classifies, route_request tool routes.
  Custom Eval 3: custom functionality #1 = the recovery prompt (category
                 definitions) that targets the Assignment 1 issue_category failure.
  Custom Eval 4: custom functionality #2 = the guardrail wrapper (measured fully
                 in Part D; here we just confirm it passes benign triage through).

    python run_partC.py --model-path "C:\\...\\phi4-mini-instruct\\phi4-mini-instruct"
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import scenario
import task
import triage_models
import triage_tools
from triage_api import TriageAPI
from guardrail import GuardedTriageAPI


def load_split(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def run_structured(api, rows):
    schema_text = json.dumps(triage_models.TriageResult.model_json_schema())
    out, scored = [], []
    for i, r in enumerate(rows, 1):
        resp = api.structured(task.user_prompt(r["input"]), schema_text, system_prompt=task.SYSTEM_V1)
        pred = task.parse(resp.text)
        ok, _, err = triage_models.validate_reply(pred)
        s = task.score_one(pred, r["gold"])
        s["pydantic_valid"] = ok
        scored.append(s)
        out.append({"doc_id": r["doc_id"], "pred": pred, "gold": r["gold"], "score": s,
                    "pydantic_valid": ok, "pydantic_error": err, "raw": resp.text,
                    "latency_s": round(resp.latency_s, 2),
                    "completion_tokens": resp.completion_tokens,
                    "prompt_tokens": resp.prompt_tokens})
        print(f"  [structured] {i:>2}/{len(rows)} {r['doc_id']} valid={ok} "
              f"{'OK' if s['category_correct'] else 'x'}")
    m = task.aggregate(scored)
    m["pydantic_valid_rate"] = round(sum(1 for s in scored if s["pydantic_valid"]) / len(scored), 4)
    m["mean_latency_s"] = round(sum(o["latency_s"] for o in out) / len(out), 2)
    m["mean_completion_tokens"] = round(sum(o["completion_tokens"] for o in out) / len(out), 1)
    return out, m


def run_tool(api, rows):
    """Model classifies via generate; route_request tool owns the routing.
    We score the TOOL's department against gold (the organization's routing
    policy), and the model's category against gold."""
    out, scored = [], []
    for i, r in enumerate(rows, 1):
        resp = api.generate(task.user_prompt(r["input"]), system_prompt=task.SYSTEM_V1)
        pred = task.parse(resp.text)
        routed = triage_tools.route_request(pred.get("issue_category", ""))
        combined = {"issue_category": pred.get("issue_category", ""),
                    "responsible_department": routed["responsible_department"]}
        s = task.score_one(combined, r["gold"])
        s["tool_routed"] = routed["routed"]
        scored.append(s)
        out.append({"doc_id": r["doc_id"], "model_pred": pred, "tool_result": routed,
                    "combined": combined, "gold": r["gold"], "score": s,
                    "latency_s": round(resp.latency_s, 2),
                    "completion_tokens": resp.completion_tokens,
                    "prompt_tokens": resp.prompt_tokens})
        print(f"  [tool] {i:>2}/{len(rows)} {r['doc_id']} -> {routed['responsible_department']} "
              f"{'OK' if s['department_correct'] else 'x'}")
    m = task.aggregate(scored)
    m["mean_latency_s"] = round(sum(o["latency_s"] for o in out) / len(out), 2)
    return out, m


def run_recovery(api, rows, system_prompt, label):
    out, scored = [], []
    for i, r in enumerate(rows, 1):
        resp = api.generate(task.user_prompt(r["input"]), system_prompt=system_prompt)
        pred = task.parse(resp.text)
        s = task.score_one(pred, r["gold"])
        scored.append(s)
        out.append({"doc_id": r["doc_id"], "pred": pred, "gold": r["gold"], "score": s,
                    "raw": resp.text, "latency_s": round(resp.latency_s, 2),
                    "completion_tokens": resp.completion_tokens,
                    "prompt_tokens": resp.prompt_tokens})
        print(f"  [{label}] {i:>2}/{len(rows)} {r['doc_id']} "
              f"cat={s['pred_category'] or '?'} {'OK' if s['category_correct'] else 'x'}")
    m = task.aggregate(scored)
    m["mean_latency_s"] = round(sum(o["latency_s"] for o in out) / len(out), 2)
    return out, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()
    scenario.ensure_dirs()
    triage_models.export_schema()

    rows = load_split(scenario.EVAL_PATH)[: args.n]
    api = TriageAPI(args.model_path, temperature=0.0, max_new_tokens=128)

    results = {}

    print("\n-- Custom Eval 1: structured_output + Pydantic --")
    o1, m1 = run_structured(api, rows)
    results["eval1_structured"] = m1

    print("\n-- Custom Eval 2: tool_calling (route_request) --")
    o2, m2 = run_tool(api, rows)
    results["eval2_tool"] = m2

    print("\n-- Custom Eval 3: recovery prompt (category definitions) --")
    # before/after on the SAME eval rows: v1 prompt vs v2 recovery prompt
    o3a, m3a = run_recovery(api, rows, task.SYSTEM_V1, "before")
    o3b, m3b = run_recovery(api, rows, task.SYSTEM_V2, "after")
    results["eval3_recovery_before"] = m3a
    results["eval3_recovery_after"] = m3b

    print("\n-- Custom Eval 4: guardrail passes benign triage through --")
    guarded = GuardedTriageAPI(api, log_path=scenario.OUT_DIR / "partC_guard_log.jsonl")
    passed = 0
    o4 = []
    for r in rows:
        gr = guarded.answer(task.user_prompt(r["input"]), system_prompt=task.SYSTEM_V2)
        passed += int(gr.allowed)
        o4.append({"doc_id": r["doc_id"], "allowed": gr.allowed, "category": gr.decision.category})
    results["eval4_guardrail_benign_pass_rate"] = round(passed / len(rows), 4)

    for name, rowset in [("partC_eval1_structured", o1), ("partC_eval2_tool", o2),
                         ("partC_eval3_before", o3a), ("partC_eval3_after", o3b),
                         ("partC_eval4_guard", o4)]:
        (scenario.OUT_DIR / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in rowset), encoding="utf-8")
    (scenario.OUT_DIR / "partC_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n=== PART C SUMMARY ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
