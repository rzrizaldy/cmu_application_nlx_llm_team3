"""Build the LLMBox batch input files from the split.

Each row is {id, instruction, record, meta}. `record` is the masked,
model-visible content (raw_text + table_json). `meta` is kept for scoring
and is never sent to the model: LLMBox's compose_user_message only reads
instruction/record/prompt.

Outputs
  data/inputs/{dev50,eval50,dev_all,eval_pool}_task.jsonl   instruction = A1 BASELINE + JSON format line
  data/mask_audit.json                                      masks applied per record (data-boundary evidence)
"""
from collections import Counter

from a2common import BASELINE, DATA, JSON_FORMAT_LINE, content, corpus, mask, save_json, save_jsonl, split_manifest


def main():
    split = split_manifest()
    records = {r["doc_id"]: r for r in corpus()}
    audit, totals = {}, Counter()
    masked = {}
    for doc_id, r in records.items():
        text, counts = mask(content(r))
        masked[doc_id] = text
        if counts:
            audit[doc_id] = counts
            totals.update(counts)

    instruction = BASELINE + JSON_FORMAT_LINE
    for name, ids in [("dev50", split["dev50"]), ("eval50", split["eval50"]),
                      ("dev_all", split["dev"]), ("eval_pool", split["eval_pool_for_part_d"])]:
        rows = [{"id": d, "instruction": instruction, "record": masked[d],
                 "meta": {"split": name, "source_kind": records[d]["metadata"]["source_kind"]}} for d in ids]
        save_jsonl(DATA / "inputs" / f"{name}_task.jsonl", rows)
        print(name, len(rows))

    save_json(DATA / "mask_audit.json", {
        "records_scanned": len(records),
        "records_with_masks": len(audit),
        "masks_by_category": dict(totals),
        "per_record": audit,
        "note": "Regex masking of PII/FIN spans before any text reaches the model. "
                "PHI/CONF have no regex rule; the corpus excluded them at collection time.",
    })
    print("masks:", dict(totals), "in", len(audit), "records")


if __name__ == "__main__":
    main()
