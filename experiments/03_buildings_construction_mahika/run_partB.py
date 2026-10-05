"""
run_partB.py -- Part B: baseline LLMBox experiments in mode=generate.

Runs the SAME 50 development inputs through Phi twice with different generation
hyperparameters, saves every response, and reports metrics. This is the
"baseline API" (plain generate mode, v1 prompt, no structured_output, no
guardrail) that later parts are compared against.

    python run_partB.py --model-path "C:\\...\\phi4-mini-instruct\\phi4-mini-instruct"

Writes to out/:
  partB_eval1.jsonl, partB_eval1_metrics.json   (greedy: temp 0.0)
  partB_eval3.jsonl, partB_eval3_metrics.json   (sampled: temp 0.9)
  partB_summary.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import scenario
import task
from triage_api import TriageAPI


def load_split(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def run_config(api: TriageAPI, rows: list[dict], label: str, gen: dict) -> dict:
    api.set_generation(**gen)
    out_rows, scored = [], []
    for i, r in enumerate(rows, 1):
        resp = api.generate(task.user_prompt(r["input"]), system_prompt=task.SYSTEM_V1)
        pred = task.parse(resp.text)
        s = task.score_one(pred, r["gold"])
        scored.append(s)
        out_rows.append({
            "doc_id": r["doc_id"], "pred": pred, "gold": r["gold"], "score": s,
            "raw": resp.text, "latency_s": round(resp.latency_s, 2),
            "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens,
        })
        print(f"  [{label}] {i:>2}/{len(rows)} {r['doc_id']} "
              f"cat={s['pred_category'] or '?'} {'OK' if s['category_correct'] else 'x'}")
    metrics = task.aggregate(scored)
    lat = [o["latency_s"] for o in out_rows]
    tok = [o["completion_tokens"] for o in out_rows]
    metrics["generation"] = gen
    metrics["mean_latency_s"] = round(sum(lat) / len(lat), 2)
    metrics["mean_completion_tokens"] = round(sum(tok) / len(tok), 1)
    metrics["mean_prompt_tokens"] = round(sum(o["prompt_tokens"] for o in out_rows) / len(out_rows), 1)
    return {"rows": out_rows, "metrics": metrics}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()
    scenario.ensure_dirs()

    rows = load_split(scenario.DEV_PATH)[: args.n]
    api = TriageAPI(args.model_path, temperature=0.0, max_new_tokens=128)

    # Baseline Evaluation 1: deterministic / greedy decoding.
    ev1 = run_config(api, rows, "eval1", {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 128})
    # Baseline Evaluation 3: higher temperature + nucleus sampling 
    
    ev3 = run_config(api, rows, "eval3", {"temperature": 0.9, "top_p": 0.95, "max_new_tokens": 128})

    for name, res in [("partB_eval1", ev1), ("partB_eval3", ev3)]:
        (scenario.OUT_DIR / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in res["rows"]), encoding="utf-8")
        (scenario.OUT_DIR / f"{name}_metrics.json").write_text(
            json.dumps(res["metrics"], indent=2), encoding="utf-8")

    summary = {"eval1_greedy": ev1["metrics"], "eval3_sampled": ev3["metrics"]}
    (scenario.OUT_DIR / "partB_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== PART B SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
