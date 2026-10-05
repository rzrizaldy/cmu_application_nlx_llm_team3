"""Score an LLMBox batch run against the A2 human reference labels.

  python score.py <run_name> [<run_name> ...]      -> runs/<run>/metrics.json, runs/<run>/scored.jsonl
  python score.py --draft <run_name> ...           -> separate metrics_draft.json, scored_draft.jsonl
  python score.py --determinism <run_name>         -> share of rows whose output is identical across repeats

Scoring follows A1 compare() (extraction.py:164): an output that fails the
schema counts as wrong on every field, and responsible_department is compared
as a set. Added in A2:
  * Cohen's kappa per categorical field, via LLMBox src/evaluator.py AgreementMeasures
  * macro-F1 per categorical field
  * department precision/recall over names, and the *ungrounded* rate:
    predicted names that do not appear verbatim in the record (the label rule
    says names must be copied from the input), plus the share outside the
    311 codebook's department list
  * truncation, latency and token statistics
  * bootstrap 95% confidence intervals (2,000 resamples, seed 952)
"""
import csv
import json
import random
import statistics
import sys
from pathlib import Path

import jsonschema

from a2common import CODEBOOK, FIELDS, LABELS, RUNS, SCHEMA, all_labels, content, corpus, load_jsonl, save_json, save_jsonl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "llmbox"))
from src.evaluator import AgreementMeasures  # noqa: E402  (LLMBox's own agreement measures)

CATEGORICAL = ["service_focus", "information_type"]


def parse_json(text):
    """Pull the first JSON object out of free text (handles ``` fences and chatter)."""
    if text is None:
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


def codebook_departments():
    with open(CODEBOOK, encoding="utf-8") as f:
        return {row["department"].strip() for row in csv.DictReader(f) if row["department"].strip()}


def macro_f1(gold, pred):
    labels = sorted(set(gold))
    f1s = []
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def bootstrap_ci(values, n=2000, seed=952):
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(k)) / k for _ in range(n))
    return [round(means[int(0.025 * n)], 4), round(means[int(0.975 * n) - 1], 4)]


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def load_run(run):
    rows = [json.loads(x) for x in (RUNS / run / "responses.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r for r in rows if r.get("repeat", 0) == 0]


def score(run, draft=False):
    labels = ({r["doc_id"]: r["assistant_draft_label"] for r in
               load_jsonl(LABELS / "assistant_draft_labels_a2.jsonl")} if draft else all_labels())
    texts = {r["doc_id"]: content(r) for r in corpus()}
    known = codebook_departments()
    rows = load_run(run)
    missing = [r["id"] for r in rows if r["id"] not in labels]
    if missing:
        raise SystemExit(f"{run}: {len(missing)} rows have no human label yet (e.g. {missing[:3]})")

    per = []
    for r in rows:
        gold = labels[r["id"]]
        pred = r.get("parsed") if r.get("parsed") is not None else parse_json(r.get("text"))
        blocked = bool(r.get("blocked"))
        try:
            jsonschema.validate(pred, SCHEMA)
            valid = not blocked
        except jsonschema.ValidationError:
            valid = False
        correct = {f: bool(valid and (set(pred[f]) == set(gold[f]) if f == "responsible_department" else pred[f] == gold[f]))
                   for f in FIELDS}
        pred_depts = list(pred.get("responsible_department") or []) if isinstance(pred, dict) and isinstance(pred.get("responsible_department"), list) else []
        pred_depts = [d for d in pred_depts if isinstance(d, str)]
        per.append({
            "id": r["id"], "kind": r.get("meta", {}).get("source_kind"),
            "parsed_json": pred is not None, "valid": valid, "blocked": blocked,
            "field_correct": correct, "exact": all(correct.values()),
            "gold": gold, "pred": pred,
            "pred_departments": pred_depts,
            "ungrounded_departments": [d for d in pred_depts if d not in texts[r["id"]]],
            "non_codebook_departments": [d for d in pred_depts if d not in known],
            "finish_reason": r.get("finish_reason"),
            "latency_s": r.get("latency_s"), "prompt_tokens": r.get("prompt_tokens"),
            "completion_tokens": r.get("completion_tokens"), "retries": r.get("retries", 0),
            "tool_calls": r.get("tool_calls"),
        })

    n = len(per)
    metrics = {"run": run, "n": n, "reference_origin":
               "ai_assisted_draft_pending_student_review" if draft else "student_reviewed_reference"}
    metrics["json_parse_rate"] = sum(p["parsed_json"] for p in per) / n
    metrics["valid_output_rate"] = sum(p["valid"] for p in per) / n
    metrics["exact_match"] = sum(p["exact"] for p in per) / n
    metrics["exact_match_ci95"] = bootstrap_ci([int(p["exact"]) for p in per])
    metrics["field_accuracy"] = {f: sum(p["field_correct"][f] for p in per) / n for f in FIELDS}
    metrics["field_accuracy_ci95"] = {f: bootstrap_ci([int(p["field_correct"][f]) for p in per]) for f in FIELDS}
    metrics["kappa"], metrics["macro_f1"] = {}, {}
    for f in CATEGORICAL:
        gold = [p["gold"][f] for p in per]
        pred = [p["pred"][f] if p["valid"] else "INVALID" for p in per]
        metrics["kappa"][f] = round(AgreementMeasures.cohen_kappa(gold, pred), 4)
        metrics["macro_f1"][f] = round(macro_f1(gold, pred), 4)

    gold_names = sum(len(p["gold"]["responsible_department"]) for p in per)
    pred_names = sum(len(p["pred_departments"]) for p in per)
    hit = sum(len(set(p["pred_departments"]) & set(p["gold"]["responsible_department"])) for p in per)
    metrics["department"] = {
        "precision": hit / pred_names if pred_names else None,
        "recall": hit / gold_names if gold_names else None,
        "predicted_names": pred_names,
        "ungrounded_name_rate": sum(len(p["ungrounded_departments"]) for p in per) / pred_names if pred_names else 0.0,
        "records_with_ungrounded_name": sum(bool(p["ungrounded_departments"]) for p in per) / n,
        "non_codebook_name_rate": sum(len(p["non_codebook_departments"]) for p in per) / pred_names if pred_names else 0.0,
    }
    lat = [p["latency_s"] for p in per if p["latency_s"] is not None]
    pt = [p["prompt_tokens"] for p in per if p["prompt_tokens"] is not None]
    ct = [p["completion_tokens"] for p in per if p["completion_tokens"] is not None]
    metrics["truncation_rate"] = sum(p["finish_reason"] == "length" for p in per) / n
    metrics["blocked_rate"] = sum(p["blocked"] for p in per) / n
    metrics["mean_retries"] = sum(p["retries"] or 0 for p in per) / n
    if lat:
        metrics["latency_s"] = {"mean": round(statistics.mean(lat), 3), "p50": round(pct(lat, 0.5), 3),
                                "p95": round(pct(lat, 0.95), 3), "total": round(sum(lat), 1)}
    if pt and ct:
        metrics["tokens"] = {"prompt_mean": round(statistics.mean(pt), 1), "completion_mean": round(statistics.mean(ct), 1),
                             "completion_p99": pct(ct, 0.99), "completion_max": max(ct),
                             "total_per_request": round(statistics.mean(pt) + statistics.mean(ct), 1)}
    tool_rows = [p for p in per if p["tool_calls"] is not None]
    if tool_rows:
        calls = [c for p in tool_rows for c in p["tool_calls"]]
        malformed = [p for p in tool_rows if '"arguments"' in (next(r for r in rows if r["id"] == p["id"]).get("text") or "")]
        metrics["tools"] = {
            "rows_with_executed_call": sum(any(not c["rejected"] for c in p["tool_calls"]) for p in tool_rows) / len(tool_rows),
            "calls_total": len(calls),
            "calls_rejected": sum(bool(c["rejected"]) for c in calls),
            "calls_by_name": {n: sum(c["name"] == n for c in calls) for n in sorted({str(c["name"]) for c in calls})},
            "queue_writes": sum(c["name"] == "queue_tag_update" and not c["rejected"] and isinstance(c["result"], dict)
                                and c["result"].get("queued") for c in calls),
            "rows_final_text_is_unexecuted_call": len(malformed) / len(tool_rows),
            "rows_nudged": sum(bool(next(r for r in rows if r["id"] == p["id"]).get("nudged")) for p in tool_rows) / len(tool_rows),
        }
    metrics["by_kind_exact"] = {k: round(sum(p["exact"] for p in per if p["kind"] == k) / max(1, sum(p["kind"] == k for p in per)), 4)
                                for k in sorted({p["kind"] for p in per})}
    save_json(RUNS / run / ("metrics_draft.json" if draft else "metrics.json"), metrics)
    save_jsonl(RUNS / run / ("scored_draft.jsonl" if draft else "scored.jsonl"), per)
    return metrics


def determinism(run):
    rows = [json.loads(x) for x in (RUNS / run / "responses.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    by_id = {}
    for r in rows:
        by_id.setdefault(r["id"], {})[r["repeat"]] = r["text"]
    pairs = [v for v in by_id.values() if 0 in v and 1 in v]
    same = sum(v[0] == v[1] for v in pairs)
    out = {"run": run, "rows": len(pairs), "identical_across_repeats": same, "identical_rate": same / len(pairs) if pairs else None}
    save_json(RUNS / run / "determinism.json", out)
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--determinism":
        for run in args[1:]:
            print(determinism(run))
    else:
        draft = bool(args and args[0] == "--draft")
        for run in (args[1:] if draft else args):
            m = score(run, draft=draft)
            print(json.dumps({k: m[k] for k in ["run", "n", "valid_output_rate", "exact_match", "field_accuracy"]}, indent=1))
