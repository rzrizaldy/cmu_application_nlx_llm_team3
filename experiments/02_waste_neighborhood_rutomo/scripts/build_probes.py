"""Build the Part D input file from Rizaldy's probe seeds (data/probes/probes_seed.jsonl).

* `record` probes are planted in the middle of a *copy* of an EVAL-pool record.
* `{CANARY}` in any probe text is replaced with one random token per build, stored in
  data/probes/canary.json. The guardrail is never given this token.
* `--benign-records N` (default 20) adds N untouched EVAL-pool records as benign tagging
  requests, so over-refusal is measured on the real workload as well as on written requests.
"""
import argparse
import json
import random
import secrets

from a2common import BASELINE, DATA, JSON_FORMAT_LINE, load_jsonl, save_json, save_jsonl

CATS = {"toxic", "out_of_scope", "injection", "leakage", "benign"}


def main(n_benign, seed_file=None):
    seeds = load_jsonl(seed_file or DATA / "probes" / "probes_seed.jsonl")
    pool = load_jsonl(DATA / "inputs" / "eval_pool_task.jsonl")
    rng = random.Random(952)
    canary = "PGH311-CANARY-" + secrets.token_hex(3).upper()
    instruction = BASELINE + JSON_FORMAT_LINE
    rows, ids = [], set()
    for s in seeds:
        assert s["category"] in CATS and s["kind"] in {"prompt", "record"}, s
        assert s["id"] not in ids, f"duplicate id {s['id']}"
        ids.add(s["id"])
        text = s["text"].replace("{CANARY}", canary)
        meta = {"category": s["category"], "kind": s["kind"], "secret": s.get("secret"),
                "uses_canary": "{CANARY}" in s["text"]}
        if s["kind"] == "prompt":
            rows.append({"id": s["id"], "prompt": text, "meta": meta})
        else:
            base = rng.choice(pool)
            words = base["record"].split(" ")
            cut = len(words) // 2
            planted = " ".join(words[:cut] + [text] + words[cut:])
            rows.append({"id": s["id"], "instruction": instruction, "record": planted,
                         "meta": {**meta, "host_record": base["id"]}})
    for base in rng.sample(pool, min(n_benign, len(pool))):
        rows.append({"id": f"benign-rec-{base['id'][-6:]}", "instruction": instruction, "record": base["record"],
                     "meta": {"category": "benign", "kind": "record", "secret": None, "uses_canary": False,
                              "host_record": base["id"]}})
    save_jsonl(DATA / "inputs" / "partd_probes.jsonl", rows)
    save_json(DATA / "probes" / "canary.json", {"canary": canary})
    counts = {}
    for r in rows:
        counts[r["meta"]["category"]] = counts.get(r["meta"]["category"], 0) + 1
    print(f"{len(rows)} Part D rows: {counts}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benign-records", type=int, default=20)
    ap.add_argument("--seed-file", default=None)
    a = ap.parse_args()
    main(a.benign_records, a.seed_file)
