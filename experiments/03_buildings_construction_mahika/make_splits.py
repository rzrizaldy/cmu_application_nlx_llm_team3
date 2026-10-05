"""
make_splits.py -- build a reproducible dev/eval split from the Assignment 1 corpus.

The assignment asks for a development set (Part B hyperparameter tuning) and a
held-out evaluation set (Part C/D). We draw 50 records for each, stratified by
issue_category (from metadata.codebook_category) so both splits contain all four
categories, using a fixed seed so the split is reproducible.

    python make_splits.py --corpus /path/to/corpus.jsonl

By default it looks for ./data/corpus.jsonl. The gold label for each record is
metadata.codebook_category
the model never sees it -- it is only used to score the model's predictions.
"""
from __future__ import annotations
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import scenario


def load_corpus(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def gold_of(rec: dict) -> dict:
    """The reference labels the model is scored against, taken from the record."""
    meta = rec.get("metadata", {}) or {}
    table = rec.get("table_json", {}) or {}
    return {
        "issue_category": meta.get("codebook_category_normalized")
        or _normalize_cat(meta.get("codebook_category", "")),
        "responsible_department": table.get("department") or "unknown",
    }


def _normalize_cat(raw: str) -> str:
    m = {
        "Building Maintenance": "building_maintenance",
        "Construction Issues": "construction",
        "Permits": "permits",
        "Accessibility": "accessibility",
    }
    return m.get(raw, raw.strip().lower().replace(" ", "_"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(scenario.CORPUS_PATH))
    ap.add_argument("--n-dev", type=int, default=50)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    scenario.ensure_dirs()
    src = Path(args.corpus)
    if not src.exists():
        raise SystemExit(
            f"Corpus not found at {src}. Copy your Assignment 1 corpus.jsonl to "
            f"{scenario.CORPUS_PATH} or pass --corpus.")
    if src.resolve() != scenario.CORPUS_PATH.resolve():
        scenario.CORPUS_PATH.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    rows = load_corpus(scenario.CORPUS_PATH)
    rng = random.Random(args.seed)

    # stratify by gold issue_category
    by_cat: dict[str, list] = defaultdict(list)
    for r in rows:
        by_cat[gold_of(r)["issue_category"]].append(r)
    for c in by_cat:
        rng.shuffle(by_cat[c])

    def take(n: int, pool: dict, used: set) -> list:
        cats = scenario.ISSUE_CATEGORIES
        per = max(1, n // len(cats))
        picked = []
        for c in cats:
            avail = [r for r in pool.get(c, []) if r["doc_id"] not in used]
            for r in avail[:per]:
                picked.append(r); used.add(r["doc_id"])
        # top up to exactly n from anything left
        leftovers = [r for r in rows if r["doc_id"] not in used]
        rng.shuffle(leftovers)
        while len(picked) < n and leftovers:
            r = leftovers.pop()
            picked.append(r); used.add(r["doc_id"])
        return picked[:n]

    used: set = set()
    dev = take(args.n_dev, by_cat, used)
    evl = take(args.n_eval, by_cat, used)

    def dump(recs: list, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps({
                    "doc_id": r["doc_id"],
                    "input": r.get(scenario.INPUT_FIELD, ""),
                    "gold": gold_of(r),
                }, ensure_ascii=False) + "\n")

    dump(dev, scenario.DEV_PATH)
    dump(evl, scenario.EVAL_PATH)

    def dist(recs):
        d = defaultdict(int)
        for r in recs:
            d[gold_of(r)["issue_category"]] += 1
        return dict(d)

    print(f"dev  -> {scenario.DEV_PATH.name}: {len(dev)} records {dist(dev)}")
    print(f"eval -> {scenario.EVAL_PATH.name}: {len(evl)} records {dist(evl)}")
    print(f"overlap: {len(set(r['doc_id'] for r in dev) & set(r['doc_id'] for r in evl))} (should be 0)")


if __name__ == "__main__":
    main()
