"""Split the 180-record A1 corpus into DEV (90) and EVAL (90), stratified by
source_kind, then draw the 50-record run samples DEV-50 and EVAL-50.

* All 25 records of the A1 evaluation sample go to DEV. The A1 recovery
  prompt was tuned on them, so they must not be used to measure anything final.
* DEV-50 = those 25 records + 25 more DEV records. It is used for Part B
  (baseline hyper-parameter work). All 100 DEV-50/EVAL-50 records are
  labelled fresh for A2 (see export_labeling.py).
* EVAL-50 is used once per Part C configuration. The other 40 EVAL records
  are the pool for Part D benign inputs.
* Per-kind targets use largest-remainder rounding of the corpus proportions.

Run once. The script refuses to overwrite an existing split unless --force is given.
"""
import argparse
import random
from collections import Counter

from a2common import DATA, CORPUS, A1_LABELS, KINDS, corpus, file_sha, load_jsonl, save_json

SEED = 952  # course number 95-820, A2


def targets(counts, total):
    n = sum(counts.values())
    raw = {k: counts[k] * total / n for k in counts}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: total - sum(out.values())]:
        out[k] += 1
    return out


def main(force=False):
    path = DATA / "split_manifest.json"
    if path.exists() and not force:
        raise SystemExit(f"{path} exists; pass --force to rebuild (this changes every downstream run).")
    records = corpus()
    kind = {r["doc_id"]: r["metadata"]["source_kind"] for r in records}
    a1_ids = [r["doc_id"] for r in load_jsonl(A1_LABELS)]
    rng = random.Random(SEED)

    by_kind = {k: sorted(d for d in kind if kind[d] == k) for k in KINDS}
    dev_target = targets({k: len(v) for k, v in by_kind.items()}, 90)
    dev, eval_ = [], []
    for k in KINDS:
        forced = [d for d in by_kind[k] if d in a1_ids]
        rest = [d for d in by_kind[k] if d not in a1_ids]
        rng.shuffle(rest)
        need = dev_target[k] - len(forced)
        assert need >= 0, f"more A1 records of kind {k} than the DEV quota"
        dev += forced + rest[:need]
        eval_ += rest[need:]

    sample_target = targets(Counter(kind[d] for d in dev), 50)
    dev50 = list(a1_ids)
    for k in KINDS:
        have = sum(kind[d] == k for d in dev50)
        pool = sorted(d for d in dev if kind[d] == k and d not in dev50)
        rng.shuffle(pool)
        dev50 += pool[: max(0, sample_target[k] - have)]
    if len(dev50) < 50:  # A1 over-filled a kind; top up from what is left
        pool = sorted(d for d in dev if d not in dev50)
        rng.shuffle(pool)
        dev50 += pool[: 50 - len(dev50)]

    eval_target = targets(Counter(kind[d] for d in eval_), 50)
    eval50 = []
    for k in KINDS:
        pool = sorted(d for d in eval_ if kind[d] == k)
        rng.shuffle(pool)
        eval50 += pool[: eval_target[k]]

    assert len(dev) == 90 and len(eval_) == 90 and not set(dev) & set(eval_)
    assert len(dev50) == 50 and len(eval50) == 50 and set(dev50) <= set(dev) and set(eval50) <= set(eval_)
    manifest = {
        "seed": SEED,
        "corpus_sha256": file_sha(CORPUS),
        "a1_labels_sha256": file_sha(A1_LABELS),
        "dev": sorted(dev), "eval": sorted(eval_),
        "dev50": sorted(dev50), "eval50": sorted(eval50),
        "eval_pool_for_part_d": sorted(set(eval_) - set(eval50)),
        "kind_counts": {name: dict(Counter(kind[d] for d in ids)) for name, ids in
                        [("dev", dev), ("eval", eval_), ("dev50", dev50), ("eval50", eval50)]},
    }
    save_json(path, manifest)
    for name in ["dev", "eval", "dev50", "eval50"]:
        print(name, len(manifest[name]), manifest["kind_counts"][name])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
