#!/usr/bin/env python3
"""Run one team experiment on the EVAL (or DEV) split and score it.

Every run goes through api/team311/pipeline.py Router.route_complaint, so the
experiments test the same code the API ships. T4_finetuned is the T0 prompt on
the LoRA-adapted model (--adapter).

Metrics follow the brief's proximal measures: issue and department accuracy,
completeness, clarification rate, and abstention, plus domain and category
accuracy, schema validity against Ticket311, latency, and tokens. Accuracy is
broken down by gold domain with 95% bootstrap intervals.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "api"))
sys.path.insert(0, str(REPO / "api" / "llmbox" / "src"))
from pydantic_models.ticket311 import Ticket311  # noqa: E402
from team311.guardrail import check_input  # noqa: E402
from team311.knowledge import issue_key  # noqa: E402
from team311.model import PhiRunner, phi_path  # noqa: E402
from team311.pipeline import Router  # noqa: E402

DATA = HERE / "data"
RUNS = HERE / "runs"
RUN_MODES = {
    "T0_generate": "T0_generate",
    "T1_structured_rag": "T1_structured_rag",
    "T2_tools": "T2_tools",
    "T3_guarded": "T3_guarded",
    "T4_finetuned": "T0_generate",
}
MINGCHIN_PROBES = REPO / "experiments" / "04_parks_public_spaces_mingchin" / "data" / "pittsburgh311"
PROBE_SETS = {
    "mahika_buildings": [REPO / "experiments" / "03_buildings_construction_mahika" / "out" / "probes.jsonl"],
    "rutomo_waste": [REPO / "experiments" / "02_waste_neighborhood_rutomo" / "data" / "inputs" / "partd_probes.jsonl"],
    "mingchin_parks": [MINGCHIN_PROBES / "adversarial_probes.jsonl", MINGCHIN_PROBES / "benign_safety_inputs.jsonl"],
}


def jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_split(split: str, limit: int) -> list[dict]:
    rows = jsonl(DATA / f"{split}.jsonl")
    if split == "dev":
        rng = random.Random(952)
        by_sub: dict[str, list[dict]] = {}
        for r in rows:
            by_sub.setdefault(r["subtopic_key"], []).append(r)
        per = max(1, limit // len(by_sub))
        rows = [r for sk in sorted(by_sub) for r in rng.sample(by_sub[sk], min(per, len(by_sub[sk])))]
    return rows[:limit]


def norm(s) -> str:
    return (s or "").strip().lower() if isinstance(s, str) else ""


def bootstrap_ci(values: list[float], n: int = 1000, seed: int = 952) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(k)) / k for _ in range(n))
    return [round(means[int(0.025 * n)], 4), round(means[int(0.975 * n) - 1], 4)]


def item_scores(r: dict) -> dict:
    p, g = r["pred"], r["gold"]
    try:
        Ticket311.model_validate({k: v for k, v in p.items() if k in Ticket311.model_fields})
        valid = bool(p.get("category")) and bool(p.get("department"))
    except Exception:
        valid = False
    s = {
        "schema_valid": valid,
        "domain": norm(p.get("domain")) == norm(g["domain"]),
        "category": norm(p.get("category")) == norm(g["category"]),
        "issue": bool(g.get("issue")) and issue_key(p.get("issue")) == issue_key(g["issue"]),
        "department": norm(p.get("department")) == norm(g["department"]),
        "complete": all(p.get(k) for k in ("domain", "category", "issue", "department")),
        "clarified": bool(p.get("clarification_question")),
        "abstained": bool(p.get("abstain")),
    }
    s["routed_correctly"] = s["category"] and s["department"]
    return {k: float(v) for k, v in s.items()}


def score_run(rows: list[dict]) -> dict:
    scored = [item_scores(r) for r in rows]
    metrics = {"n": len(rows)}
    for key in ("issue", "department", "routed_correctly", "domain", "category"):
        vals = [s[key] for s in scored]
        metrics[f"{key}_accuracy"] = round(statistics.mean(vals), 4) if vals else 0.0
        metrics[f"{key}_accuracy_ci95"] = bootstrap_ci(vals)
    for key, name in (("schema_valid", "schema_valid_rate"), ("complete", "completeness_rate"),
                      ("clarified", "clarification_rate"), ("abstained", "abstention_rate")):
        metrics[name] = round(statistics.mean(s[key] for s in scored), 4) if scored else 0.0
    for key in ("latency_s", "prompt_tokens", "output_tokens"):
        vals = [r[key] for r in rows if r.get(key)]
        metrics[f"mean_{key}"] = round(statistics.mean(vals), 3) if vals else None
    metrics["by_domain"] = {}
    for dom in sorted({r["gold"]["domain"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["gold"]["domain"] == dom]
        metrics["by_domain"][dom] = {"n": len(idx), **{
            k: round(statistics.mean(scored[i][k] for i in idx), 4)
            for k in ("issue", "department", "routed_correctly", "category")
        }}
    metrics["by_input_origin"] = {}
    for origin in sorted({r["gold"]["input_origin"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["gold"]["input_origin"] == origin]
        metrics["by_input_origin"][origin] = {"n": len(idx), "routed_correctly": round(
            statistics.mean(scored[i]["routed_correctly"] for i in idx), 4)}
    metrics["raw_json_rate"] = round(statistics.mean(
        1.0 if r.get("trace", {}).get("parsed") == "json" else 0.0 for r in rows), 4) if rows else 0.0
    metrics["vote_fallback_rate"] = round(statistics.mean(
        1.0 if r["pred"].get("issue_source") == "vote_fallback" else 0.0 for r in rows), 4) if rows else 0.0
    metrics["gold_from_codebook"] = sum(1 for r in rows if r["gold"].get("gold_source") == "codebook")
    return metrics


def _probe_rows(path: Path) -> list[dict]:
    rows = []
    for d in jsonl(path):
        if "expected" in d:
            rows.append({"id": d["probe_id"], "category": d["category"], "text": d["input"], "expected": d["expected"]})
        elif "meta" in d:
            meta = d["meta"]
            text = d.get("prompt") or json.dumps({k: v for k, v in d.items() if k not in ("id", "meta")}, ensure_ascii=False)
            benign = meta.get("category") == "benign" or (meta.get("category") == "leakage" and meta.get("kind") == "record")
            rows.append({"id": d["id"], "category": meta.get("category"), "text": text, "expected": "allow" if benign else "block"})
        else:
            benign = "category" not in d
            rows.append({"id": d["id"], "category": d.get("category", "benign"), "text": d["input"],
                         "expected": "allow" if benign else "block"})
    return rows


def score_probes(out_dir: Path) -> dict:
    """Input-guard decisions on the members' Part D probe sets (no generation needed)."""
    summary, log = {}, []
    for name, paths in PROBE_SETS.items():
        rows = [r for p in paths if p.exists() for r in _probe_rows(p)]
        if not rows:
            continue
        for r in rows:
            decision = "block" if check_input(r["text"]) else "allow"
            log.append({"set": name, **{k: r[k] for k in ("id", "category", "expected")},
                        "decision": decision, "correct": decision == r["expected"]})
        adv = [x for x in log if x["set"] == name and x["expected"] == "block"]
        ben = [x for x in log if x["set"] == name and x["expected"] == "allow"]
        summary[name] = {
            "n": len(rows),
            "adversarial_block_rate": round(sum(x["correct"] for x in adv) / max(len(adv), 1), 4),
            "benign_pass_rate": round(sum(x["correct"] for x in ben) / max(len(ben), 1), 4) if ben else None,
        }
    (out_dir / "probe_decisions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=list(RUN_MODES))
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--split", choices=["eval", "dev"], default="eval")
    args = ap.parse_args()

    dev = jsonl(DATA / "dev.jsonl")
    rows = load_split(args.split, args.limit)
    if args.split == "dev":
        held = {r["doc_id"] for r in rows}
        dev = [d for d in dev if d["doc_id"] not in held]
    router = Router(PhiRunner(phi_path(), args.adapter), mode=RUN_MODES[args.run], dev_examples=dev)

    out_dir = (RUNS if args.split == "eval" else RUNS / "dev") / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for item in rows:
        out = router.route_complaint(item["input"])
        rec = {**item, "pred": out["ticket"], "raw": out["raw"], "trace": out["trace"],
               "latency_s": out["latency_s"], "prompt_tokens": out["prompt_tokens"], "output_tokens": out["output_tokens"]}
        results.append(rec)
        print(f"  {item['doc_id']}: {rec['pred'].get('issue')} | gold {item['gold']['issue']}", flush=True)

    metrics = {"run": args.run, "mode": RUN_MODES[args.run], "split": args.split, **score_run(results),
               "model_path": str(phi_path()), "adapter": str(args.adapter) if args.adapter else None}
    if args.run == "T3_guarded":
        metrics["guardrail_probes"] = score_probes(out_dir)
    (out_dir / "responses.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
