"""
compute_costbenefit.py -- Part E: organizational cost/benefit from the real
numbers produced by Parts B, C and D. No model needed; reads out/*.json.

Cost side: tokens + latency per request (B/C), guardrail overhead (D),
estimated human-review time implied by the Part A policy.
Benefit side: how many outputs are usable without correction (from accuracy),
cost of correcting the rest, vs. the manual process and a non-LLM baseline.
Break-even: the request volume at which the API is cheaper than manual triage.

Edit the ASSUMPTIONS block to match the numbers you decide to defend in the
memo -- they are yours to set, and each is labeled so you can cite it.

    python compute_costbenefit.py
"""
from __future__ import annotations
import json
from pathlib import Path

import scenario


ASSUMPTIONS = {
    "clerk_hourly_usd": 25.0,          # loaded cost of a 311 clerk
    "manual_seconds_per_request": 90,  # time to read + classify + route by hand
    "review_seconds_per_flagged": 30,  # human review when the policy requires it
    "share_requiring_review": 0.30,    # from Part A HITL line (low-confidence/abstain)
    "correction_seconds": 60,          # time to fix one wrong API output
    # non-LLM baseline: a keyword rule over request_type. Near-zero compute,
    # but only works when the text literally contains the category word.
    "keyword_baseline_accuracy": 0.70, # set from your own quick check / estimate
}


def load(name, default=None):
    p = scenario.OUT_DIR / name
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    A = ASSUMPTIONS
    partB = load("partB_summary.json", {})
    partC = load("partC_summary.json", {})
    partD = load("partD_report.json", {})

    # --- usable-without-correction rate: best category accuracy we achieved ---
    cand = []
    if partC:
        if "eval3_recovery_after" in partC:
            cand.append(partC["eval3_recovery_after"].get("category_accuracy", 0))
        if "eval1_structured" in partC:
            cand.append(partC["eval1_structured"].get("category_accuracy", 0))
    if partB:
        cand.append(partB.get("eval1_greedy", {}).get("category_accuracy", 0))
    usable = max(cand) if cand else 0.0

    # --- per-request cost in seconds (API path) ---
    api_latency = 0.0
    if partD and partD.get("cost"):
        api_latency = partD["cost"].get("mean_guarded_latency_s", 0)
    elif partB:
        api_latency = partB.get("eval1_greedy", {}).get("mean_latency_s", 0)

    review_s = A["share_requiring_review"] * A["review_seconds_per_flagged"]
    correction_s = (1 - usable) * A["correction_seconds"]
    api_human_s = review_s + correction_s          # human time per request on API path
    manual_s = A["manual_seconds_per_request"]     # human time per request, manual

    sec_to_cost = A["clerk_hourly_usd"] / 3600.0
    api_cost = api_human_s * sec_to_cost           # compute cost ~ 0 (local model)
    manual_cost = manual_s * sec_to_cost

    saved_per_request = manual_cost - api_cost

    report = {
        "inputs_used": {
            "usable_without_correction_rate": round(usable, 3),
            "api_latency_s_per_request": round(api_latency, 2),
            "assumptions": A,
        },
        "per_request_seconds": {
            "manual": manual_s,
            "api_human_review": round(api_human_s, 1),
            "api_wall_clock_model": round(api_latency, 2),
        },
        "per_request_cost_usd": {
            "manual": round(manual_cost, 4),
            "api": round(api_cost, 4),
            "saved": round(saved_per_request, 4),
        },
        "alternatives": {
            "manual_only": {"accuracy": "~1.0 (human)", "cost_per_req_usd": round(manual_cost, 4)},
            "keyword_baseline": {"accuracy": A["keyword_baseline_accuracy"],
                                 "cost_per_req_usd": 0.0,
                                 "note": "near-zero compute; misses anything not literally named"},
            "llm_api": {"accuracy": round(usable, 3), "cost_per_req_usd": round(api_cost, 4)},
        },
    }

    # break-even volume: if there is a fixed setup cost, how many requests until
    # per-request savings pay it back. Set your own setup estimate here.
    setup_hours = 20
    setup_cost = setup_hours * A["clerk_hourly_usd"]
    if saved_per_request > 0:
        report["break_even_requests"] = round(setup_cost / saved_per_request)
        report["break_even_note"] = (
            f"At ${saved_per_request:.4f} saved per request, the ~${setup_cost:.0f} "
            f"setup pays back after ~{report['break_even_requests']:,} requests.")
    else:
        report["break_even_requests"] = None
        report["break_even_note"] = (
            "API does not save human time per request under current assumptions; "
            "revisit accuracy or review share.")

    (scenario.OUT_DIR / "partE_costbenefit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== PART E COST / BENEFIT ===")
    print(json.dumps(report, indent=2))
    print("\nNote: edit ASSUMPTIONS at the top to the values you will defend; rerun to update.")


if __name__ == "__main__":
    main()
