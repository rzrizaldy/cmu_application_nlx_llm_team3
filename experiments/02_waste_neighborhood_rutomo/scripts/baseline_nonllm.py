"""Part E non-LLM alternative: TF-IDF + logistic regression for the two categorical
fields, and verbatim string matching for departments. Trained on DEV-50 human labels,
evaluated on EVAL-50. The output has the same shape as an LLMBox batch run, so
score.py scores it identically.

  python baseline_nonllm.py        -> runs/E_nonllm_tfidf_lr/{responses.jsonl, manifest.json}
  python baseline_nonllm.py --draft -> separate draft-trained run
  python score.py E_nonllm_tfidf_lr

The department gazetteer is built from (a) names in the DEV labels and (b) the 311
codebook department list. A department is predicted when its name appears verbatim
in the record. No EVAL label is used anywhere.
"""
import csv
import json
import time
import sys
from datetime import datetime, timezone

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from a2common import CODEBOOK, DATA, LABELS, RUNS, all_labels, file_sha, load_jsonl, save_json, split_manifest

RUN = "E_nonllm_tfidf_lr"


def main(draft=False):
    split = split_manifest()
    labels = ({r["doc_id"]: r["assistant_draft_label"] for r in
               load_jsonl(LABELS / "assistant_draft_labels_a2.jsonl")} if draft else all_labels())
    run_name = RUN + ("_drafttrained" if draft else "")
    dev = {r["id"]: r for r in load_jsonl(DATA / "inputs" / "dev50_task.jsonl")}
    ev = load_jsonl(DATA / "inputs" / "eval50_task.jsonl")
    train_ids = sorted(split["dev50"])
    assert not set(train_ids) & set(split["eval"])
    texts = [dev[i]["record"] for i in train_ids]

    t0 = time.perf_counter()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X = vec.fit_transform(texts)
    models = {}
    for field in ["service_focus", "information_type"]:
        y = [labels[i][field] for i in train_ids]
        models[field] = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, y)
    with open(CODEBOOK, encoding="utf-8") as f:
        gazetteer = {r["department"].strip() for r in csv.DictReader(f) if r["department"].strip()}
    gazetteer |= {d for i in train_ids for d in labels[i]["responsible_department"]}
    train_s = time.perf_counter() - t0

    out = RUNS / run_name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "responses.jsonl").open("w", encoding="utf-8") as f:
        for row in ev:
            s = time.perf_counter()
            x = vec.transform([row["record"]])
            found = sorted(d for d in gazetteer if d in row["record"])
            # drop names that are substrings of a longer matched name (e.g. "DPW" inside "DPW - Refuse")
            found = [d for d in found if not any(d != o and d in o for o in found)]
            parsed = {"service_focus": models["service_focus"].predict(x)[0],
                      "information_type": models["information_type"].predict(x)[0],
                      "responsible_department": found}
            f.write(json.dumps({"id": row["id"], "repeat": 0, "submode": "nonllm", "text": json.dumps(parsed),
                                "parsed": parsed, "prompt_tokens": 0, "completion_tokens": 0,
                                "latency_s": round(time.perf_counter() - s, 5), "finish_reason": "stop",
                                "meta": row.get("meta", {})}) + "\n")
    save_json(out / "manifest.json", {
        "run_name": run_name, "reference_origin": "ai_assisted_draft_pending_student_review" if draft else "student_reviewed_reference",
        "method": "TF-IDF(1-2gram) + LogisticRegression(balanced) per categorical field; "
                                   "verbatim gazetteer match for departments",
        "train_ids": train_ids, "train_seconds": round(train_s, 3), "gazetteer_size": len(gazetteer),
        "input_sha256": file_sha(DATA / "inputs" / "eval50_task.jsonl"),
        "finished_at": datetime.now(timezone.utc).isoformat()})
    print(f"wrote {out} (train {train_s:.2f}s, gazetteer {len(gazetteer)} names)")


if __name__ == "__main__":
    main(draft=sys.argv[1:2] == ["--draft"])
