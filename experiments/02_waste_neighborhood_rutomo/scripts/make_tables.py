"""Collect every measured result into markdown tables and an evidence packet.

  python make_tables.py   -> results/tables/*.md and memo/evidence_packet.md

Only numbers and file paths are generated here. There is no interpretation;
the memo text is written by the student. Missing results are skipped, so
this can be rerun after every new run.
"""
import json
from pathlib import Path

from a2common import A2, RESULTS, RUNS

TABLES = RESULTS / "tables"
MEMO = A2.parent / "memo"


def load(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def pct(x):
    return "—" if x is None else f"{100 * x:.0f}%"


def ci(c):
    return "—" if not c else f"[{100 * c[0]:.0f}, {100 * c[1]:.0f}]"


def settings(run):
    m = load(RUNS / run / "manifest.json") or {}
    g = m.get("generation", {})
    if not g:
        return m.get("method", "—")[:40]
    return f"T={g.get('temperature')}, top_p={g.get('top_p')}, max_new={g.get('max_new_tokens')}, sample={g.get('do_sample')}"


def run_table(runs, title, metrics_file="metrics.json"):
    lines = [f"### {title}", "",
             "| Run | Settings | n | Valid | Exact [95% CI] | Focus | Type | Dept | κ focus | κ type | Ungrounded dept names | Truncated | Latency mean / p95 (s) | Tokens in / out |",
             "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for run in runs:
        m = load(RUNS / run / metrics_file)
        if not m:
            continue
        fa, lat, tok = m["field_accuracy"], m.get("latency_s", {}), m.get("tokens", {})
        lines.append(f"| {run} | {settings(run)} | {m['n']} | {pct(m['valid_output_rate'])} | {pct(m['exact_match'])} {ci(m['exact_match_ci95'])} "
                     f"| {pct(fa['service_focus'])} | {pct(fa['information_type'])} | {pct(fa['responsible_department'])} "
                     f"| {m['kappa']['service_focus']:.2f} | {m['kappa']['information_type']:.2f} "
                     f"| {pct(m['department']['ungrounded_name_rate'])} | {pct(m['truncation_rate'])} "
                     f"| {lat.get('mean', '—')} / {lat.get('p95', '—')} | {tok.get('prompt_mean', '—')} / {tok.get('completion_mean', '—')} |")
    return "\n".join(lines) + "\n"


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    runs = sorted(p.name for p in RUNS.iterdir() if (p / "responses.jsonl").exists())
    part_b = [r for r in runs if r.startswith("B")]
    part_c = [r for r in runs if r.startswith("C") and not r.startswith("C1abl") and not r.startswith("C2abl")]
    ablations = [r for r in runs if "abl" in r]
    other = [r for r in runs if r.startswith("E_")]
    out = {}
    out["partB.md"] = run_table(part_b, "Part B: baseline generate on DEV-50")
    out["partC.md"] = run_table(part_c, "Part C: custom API on EVAL-50")
    out["ablations.md"] = run_table(ablations, "Ablations (DEV-50)")
    out["alternatives.md"] = run_table(other + part_c, "Part E: alternatives on EVAL-50")

    det = [load(RUNS / r / "determinism.json") for r in runs if (RUNS / r / "determinism.json").exists()]
    if det:
        out["determinism.md"] = "### Determinism (same rows, two repeats)\n\n| Run | Rows | Identical |\n|---|---:|---:|\n" + "".join(
            f"| {d['run']} | {d['rows']} | {d['identical_across_repeats']} ({pct(d['identical_rate'])}) |\n" for d in det)

    tool_lines = []
    for r in runs:
        m = load(RUNS / r / "metrics.json")
        if m and m.get("tools"):
            t = m["tools"]
            tool_lines.append(f"| {r} | {pct(t['rows_with_executed_call'])} | {t['calls_total']} | {t['calls_rejected']} | "
                              f"{t['queue_writes']} | {pct(t['rows_final_text_is_unexecuted_call'])} | {pct(t['rows_nudged'])} | {t['calls_by_name']} |")
    if tool_lines:
        out["tools.md"] = ("### Tool calling\n\n| Run | Rows with executed call | Calls | Rejected | Queued writes | Final text is an unexecuted call | Nudged | Calls by tool |\n"
                           "|---|---:|---:|---:|---:|---:|---:|---|\n" + "\n".join(tool_lines) + "\n")

    d = load(RESULTS / "partd_metrics.json")
    if d:
        lines = ["### Part D: baseline vs guarded", ""]
        if d.get("probe_design") and not d["probe_design"].get("human_adjudication_complete"):
            lines += ["Blind human adjudication pending before interpreting automatic probe success rules.", ""]
        lines += ["| Category | n | Baseline success | Guarded success | Catch rate | Over-refusal | Review |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for cat, e in d["by_category"].items():
            lines.append(f"| {cat} | {e['n']} | {e.get('baseline_success', '—')} | {e.get('guarded_success', '—')} | "
                         f"{pct(e.get('catch_rate'))} | {pct(e.get('over_refusal_rate'))} | {pct(e.get('review_rate'))} |")
        c = d.get("cost_on_benign_allowed") or {}
        if c:
            latency = c["added_latency_s_mean"]
            tokens = c["added_tokens_mean"]
            lines += ["", f"Guardrail cost on allowed benign rows (n={c['n']}): {latency:+} s latency, "
                          f"{tokens:+} tokens per request (baseline {c['baseline_latency_s_mean']} s, {c['baseline_tokens_mean']} tokens)."]
        else:
            lines += ["", "Guardrail cost on allowed benign rows: unavailable (none allowed)."]
        out["partD.md"] = "\n".join(lines) + "\n"
    adj = load(RESULTS / "partd_adjudication.json")
    if adj:
        out["partD_adjudication.md"] = (f"### Human adjudication of guard decisions\n\nn={adj['n']}, agreement {pct(adj['percent_agreement'])}, "
                                        f"κ={adj['cohen_kappa']}, false blocks {adj['false_blocks']}, missed blocks {adj['missed_blocks']}.\n")
    cost = load(RESULTS / "cost_model.json")
    if cost:
        lines = ["### Part E: cost model (per record)", "", f"Manual: {cost['manual_seconds_per_record']} s/record; review: {cost['review']['seconds_per_output']} s/output.", "",
                 "| Run | Usable | Usable basis | Tokens/req | Compute s | Human s with API | Manual $ | API $ | Saving $ | Break-even records |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r, e in cost["runs"].items():
            lines.append(f"| {r} | {pct(e['usable_rate'])} | {e['usable_rate_source']} | {e['tokens_per_request']} | {e['compute_seconds_per_record']} | "
                         f"{e['human_seconds_per_record_with_api']} | {e['manual_usd_per_record']} | {e['api_usd_per_record']} | "
                         f"{e['saving_usd_per_record']} | {e['break_even_records']} |")
        out["partE.md"] = "\n".join(lines) + "\n"

    for name, text in out.items():
        (TABLES / name).write_text(text, encoding="utf-8")
    MEMO.mkdir(exist_ok=True)
    packet = ["# Evidence packet (generated by a2/scripts/make_tables.py; numbers only)", ""]
    for name, text in out.items():
        packet += [f"Source table: `a2/results/tables/{name}`", "", text, ""]
    packet.append("Per-run details: `a2/runs/<run>/{manifest.json, config.yaml, responses.jsonl, metrics.json, scored.jsonl}`.")
    (MEMO / "evidence_packet.md").write_text("\n".join(packet), encoding="utf-8")
    print("tables:", sorted(out))


if __name__ == "__main__":
    main()
