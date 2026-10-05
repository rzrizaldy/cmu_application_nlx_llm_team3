#!/usr/bin/env python3
"""Run team routing experiments T0–T3 (and T4 with --adapter)."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "api"))
from team311.guardrail import check_input, redact_output  # noqa: E402
from team311.knowledge import load_knowledge, retrieve, resolution_range  # noqa: E402
from team311.tools import lookup_codebook_json  # noqa: E402

sys.path.insert(0, str(REPO / "corpora" / "05_all"))
from category_groups import CATEGORY_GROUPS  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
RUNS = Path(__file__).resolve().parent / "runs"
PHI_DEFAULT = REPO.parent / "cmu_application_of_nlx_llm" / "lab01" / "models" / "phi-4-mini-instruct"

SYSTEM = """You classify Pittsburgh 311 intake text. Output must be a single JSON object only, no prose.
Keys: domain, category, issue, department, missing_information, clarification_question, confidence, abstain, historical_resolution_range.
Format: {"domain":"<given domain>","category":"<one allowed category>","issue":"<short issue name>","department":"<city department>","missing_information":[],"clarification_question":null,"confidence":0.8,"abstain":false,"historical_resolution_range":null}"""


def parse_json(text: str) -> dict | None:
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    return None


def norm(s: str | None) -> str:
    return (s or "").strip().lower()


def bootstrap_ci(values: list[float], n: int = 500, seed: int = 952) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(k)) / k for _ in range(n))
    return [round(means[int(0.025 * n)], 4), round(means[int(0.975 * n) - 1], 4)]


class PhiRunner:
    def __init__(self, model_path: Path, adapter_path: Path | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False
        )
        if adapter_path and adapter_path.exists():
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.to(self.device)
        self.model.eval()

    def generate(self, user_content: str, max_new_tokens: int = 220) -> tuple[str, float]:
        import torch
        t0 = time.perf_counter()
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_content}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text, time.perf_counter() - t0


def build_prompt(item: dict, mode: str, kb: list[dict]) -> str | None:
    blocked = check_input(item["input"]) if mode == "T3_guarded" else None
    if blocked:
        return None
    allowed = CATEGORY_GROUPS[item["subtopic_key"]]
    parts = [
        f"domain: {item['subtopic_key']}",
        "category must be exactly one of: " + "; ".join(allowed),
        "department: the City of Pittsburgh department that should handle this request. "
        "If the text names one, copy it exactly; otherwise give your best guess from the evidence below. Do not leave it null.",
    ]
    if mode in {"T1_structured_rag", "T2_tools", "T3_guarded"}:
        docs = retrieve(item["input"], kb, subtopic_key=item["subtopic_key"], k=3)
        if docs:
            parts.append("Retrieved knowledge:\n" + "\n---\n".join(d["text"][:600] for d in docs))
    if mode in {"T2_tools", "T3_guarded"}:
        parts.append("Codebook lookup:\n" + lookup_codebook_json(item["input"], allowed))
    parts.append("Complaint:\n" + item["input"][:2000])
    return "\n\n".join(parts)


STRUCTURED_MODES = {"T1_structured_rag", "T2_tools", "T3_guarded"}


def snap_category(pred: dict, allowed: list[str]) -> dict:
    """Schema enforcement for structured modes: map a near-miss category onto the allowed list."""
    import difflib

    cat = pred.get("category")
    if not isinstance(cat, str) or cat in allowed:
        return pred
    lowered = {a.lower(): a for a in allowed}
    key = re.sub(r"[_\s]+", " ", cat).strip().lower()
    match = lowered.get(key) or next(iter(difflib.get_close_matches(key, list(lowered), n=1, cutoff=0.6)), None)
    if match:
        pred = {**pred, "category": lowered[match] if match in lowered else match, "category_raw": cat}
    return pred


def load_split(split: str, limit: int) -> list[dict]:
    rows = [json.loads(l) for l in (DATA / f"{split}.jsonl").read_text().splitlines() if l.strip()]
    if split == "dev":
        rng = random.Random(952)
        by_sub: dict[str, list[dict]] = {}
        for r in rows:
            by_sub.setdefault(r["subtopic_key"], []).append(r)
        per = max(1, limit // len(by_sub))
        rows = [r for sk in sorted(by_sub) for r in rng.sample(by_sub[sk], min(per, len(by_sub[sk])))]
    return rows[:limit]


def score_run(rows: list[dict]) -> dict:
    cat_ok = [1.0 if norm(r["pred"].get("category")) == norm(r["gold"]["category"]) else 0.0 for r in rows if r.get("pred")]
    dep_ok = [1.0 if norm(r["pred"].get("department")) == norm(r["gold"]["department"]) else 0.0 for r in rows if r.get("pred")]
    both = [1.0 if c and d else 0.0 for c, d in zip(cat_ok, dep_ok)]
    valid = [1.0 if r.get("pred") else 0.0 for r in rows]
    lat = [r["latency_s"] for r in rows if "latency_s" in r]
    by_sub = {}
    for r in rows:
        sk = r["subtopic_key"]
        by_sub.setdefault(sk, {"n": 0, "cat": 0, "dep": 0})
        by_sub[sk]["n"] += 1
        if r.get("pred") and norm(r["pred"].get("category")) == norm(r["gold"]["category"]):
            by_sub[sk]["cat"] += 1
        if r.get("pred") and norm(r["pred"].get("department")) == norm(r["gold"]["department"]):
            by_sub[sk]["dep"] += 1
    return {
        "n": len(rows),
        "category_accuracy": round(sum(cat_ok) / max(len(cat_ok), 1), 4),
        "department_accuracy": round(sum(dep_ok) / max(len(dep_ok), 1), 4),
        "both_correct": round(sum(both) / max(len(both), 1), 4),
        "schema_valid_rate": round(sum(valid) / max(len(valid), 1), 4),
        "category_accuracy_ci95": bootstrap_ci(cat_ok),
        "both_correct_ci95": bootstrap_ci(both),
        "mean_latency_s": round(statistics.mean(lat), 3) if lat else None,
        "by_subtopic": by_sub,
        "codebook_row_eval_n": sum(1 for r in rows if r.get("gold", {}).get("input_origin") == "codebook_row"),
    }


PROBE_SETS = {
    "mahika_buildings": REPO / "experiments" / "03_buildings_construction_mahika" / "out" / "probes.jsonl",
    "rutomo_waste": REPO / "experiments" / "02_waste_neighborhood_rutomo" / "data" / "inputs" / "partd_probes.jsonl",
}


def _probe_rows(name: str, path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if "expected" in d:
            rows.append({"id": d["probe_id"], "category": d["category"], "text": d["input"], "expected": d["expected"]})
            continue
        meta = d.get("meta", {})
        text = d.get("prompt") or json.dumps({k: v for k, v in d.items() if k not in ("id", "meta")}, ensure_ascii=False)
        benign_like = meta.get("category") == "benign" or (meta.get("category") == "leakage" and meta.get("kind") == "record")
        rows.append({"id": d["id"], "category": meta.get("category"), "text": text, "expected": "allow" if benign_like else "block"})
    return rows


def score_probes(out_dir: Path) -> dict:
    """Input-guard decisions on the Part D probe sets (no generation needed)."""
    summary, log = {}, []
    for name, path in PROBE_SETS.items():
        if not path.exists():
            continue
        rows = _probe_rows(name, path)
        by_cat: dict[str, dict] = {}
        for r in rows:
            decision = "block" if check_input(r["text"]) else "allow"
            ok = decision == r["expected"]
            c = by_cat.setdefault(r["category"], {"n": 0, "correct": 0})
            c["n"] += 1
            c["correct"] += int(ok)
            log.append({"set": name, **{k: r[k] for k in ("id", "category", "expected")}, "decision": decision, "correct": ok})
        adv = [x for x in log if x["set"] == name and x["expected"] == "block"]
        ben = [x for x in log if x["set"] == name and x["expected"] == "allow"]
        summary[name] = {
            "n": len(rows),
            "adversarial_block_rate": round(sum(x["correct"] for x in adv) / max(len(adv), 1), 4),
            "benign_pass_rate": round(sum(x["correct"] for x in ben) / max(len(ben), 1), 4),
            "by_category": by_cat,
        }
    (out_dir / "probe_decisions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=["T0_generate", "T1_structured_rag", "T2_tools", "T3_guarded", "T4_finetuned"])
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--split", choices=["eval", "dev"], default="eval")
    args = ap.parse_args()

    mode = args.run if args.run != "T4_finetuned" else "T0_generate"
    model_path = Path(os.environ.get("PHI_MODEL_PATH", PHI_DEFAULT))
    kb = load_knowledge()
    runner = PhiRunner(model_path, args.adapter)

    eval_rows = load_split(args.split, args.limit)
    out_dir = (RUNS if args.split == "eval" else RUNS / "dev") / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for item in eval_rows:
        prompt = build_prompt(item, mode, kb)
        rec = {**item}
        if prompt is None:
            rec["pred"] = {"status": "blocked", "abstain": True}
            rec["latency_s"] = 0.0
        else:
            text, lat = runner.generate(prompt)
            if args.run == "T3_guarded":
                text = redact_output(text)
            pred = parse_json(text) or {}
            if mode in STRUCTURED_MODES:
                pred = snap_category(pred, CATEGORY_GROUPS[item["subtopic_key"]])
            if pred.get("category") and not pred.get("historical_resolution_range"):
                pred["historical_resolution_range"] = resolution_range(
                    kb, issue=pred.get("issue"), category=pred.get("category")
                )
            rec["pred"] = pred
            rec["raw"] = text[:2000]
            rec["latency_s"] = lat
        results.append(rec)
        print(f"  {item['doc_id']} {rec.get('pred', {}).get('category', 'blocked')}")

    metrics = score_run(results)
    metrics["run"] = args.run
    metrics["split"] = args.split
    metrics["model_path"] = str(model_path)
    metrics["adapter"] = str(args.adapter) if args.adapter else None
    if args.run == "T3_guarded" and args.split == "eval":
        metrics["guardrail_probes"] = score_probes(out_dir)
    (out_dir / "responses.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
