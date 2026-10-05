"""Build the M5 few-shot pool from DEV-50 human labels, and audit a run for leakage.

  python build_fewshot_pool.py            -> data/fewshot_pool_dev.jsonl  (DEV records only)
  python build_fewshot_pool.py --draft    -> separate assistant-draft DEV pool
  python build_fewshot_pool.py --audit C4_fewshot
      -> checks that no retrieved id is an EVAL record; writes runs/<run>/leakage_audit.json
"""
import json
import sys

from a2common import DATA, LABELS, RUNS, all_labels, load_jsonl, save_json, save_jsonl, split_manifest

POOL = DATA / "fewshot_pool_dev.jsonl"


def build(draft=False):
    split = split_manifest()
    labels = ({r["doc_id"]: r["assistant_draft_label"] for r in
               load_jsonl(LABELS / "assistant_draft_labels_a2.jsonl")} if draft else all_labels())
    inputs = {r["id"]: r for r in load_jsonl(DATA / "inputs" / "dev50_task.jsonl")}
    rows = [{"id": d, "record": inputs[d]["record"], "label": labels[d]} for d in sorted(split["dev50"])]
    assert not {r["id"] for r in rows} & set(split["eval"]), "EVAL record in the few-shot pool"
    target = DATA / "fewshot_pool_assistant_draft_dev.jsonl" if draft else POOL
    save_jsonl(target, rows)
    print(f"pool: {len(rows)} labelled DEV records -> {target} ({'assistant draft' if draft else 'student reviewed'})")


def audit(run):
    split = split_manifest()
    rows = [json.loads(x) for x in (RUNS / run / "responses.jsonl").read_text().splitlines() if x.strip()]
    retrieved = [h["id"] for r in rows for h in r.get("retrieved", [])]
    leaked = sorted(set(retrieved) & set(split["eval"]))
    out = {"run": run, "rows": len(rows), "examples_shown": len(retrieved),
           "distinct_examples": len(set(retrieved)), "eval_ids_retrieved": leaked, "leak_free": not leaked}
    save_json(RUNS / run / "leakage_audit.json", out)
    print(out)


if __name__ == "__main__":
    audit(sys.argv[2]) if sys.argv[1:2] == ["--audit"] else build(draft=sys.argv[1:2] == ["--draft"])
