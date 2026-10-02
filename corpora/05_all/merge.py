#!/usr/bin/env python3
"""Merge corpora/01-04 into corpora/05_all.

Writes corpus.jsonl, human_labels.jsonl, sources.csv, taxonomy.json and
corpus_stats.json here, then refreshes appendix/original_dataset.jsonl.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from category_groups import CATEGORY_GROUPS, SUBTOPICS

ALL_DIR = Path(__file__).resolve().parent
CORPORA_DIR = ALL_DIR.parent
REPO_ROOT = CORPORA_DIR.parent
APPENDIX_ORIGINAL = REPO_ROOT / "appendix" / "original_dataset.jsonl"

REQUIRED_TOP = {
    "doc_id",
    "source_url",
    "retrieved_at",
    "modality",
    "raw_text",
    "table_json",
    "license_note",
    "metadata",
}
VALID_MODALITY = {"text", "table", "mixed"}

# The brief forbids addresses and exact coordinates in the LLM corpus. These keys
# are dropped from the team copy only; member corpora stay as submitted.
FORBIDDEN_TABLE_KEYS = {
    "address", "street_address", "address_line",
    "lat", "latitude", "lon", "lng", "longitude", "x", "y",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def content(record: dict) -> str:
    text = record.get("raw_text") or ""
    table = record.get("table_json")
    if table is not None:
        text += "\nTable:\n" + json.dumps(table, ensure_ascii=False, sort_keys=True)
    return text.strip()


def redact(table, removed: set[str]):
    if isinstance(table, dict):
        out = {}
        for key, value in table.items():
            if str(key).lower() in FORBIDDEN_TABLE_KEYS:
                removed.add(str(key))
            else:
                out[key] = redact(value, removed)
        return out
    if isinstance(table, list):
        return [redact(item, removed) for item in table]
    return table


def validate_record(record: dict, folder: str) -> None:
    doc_id = record.get("doc_id")
    missing = REQUIRED_TOP - set(record)
    extra = set(record) - REQUIRED_TOP
    if missing:
        raise ValueError(f"{folder}: missing fields {missing} on {doc_id}")
    if extra:
        raise ValueError(f"{folder}: unexpected fields {extra} on {doc_id}")
    if not record["source_url"].startswith("https://"):
        raise ValueError(f"{folder}: source_url must be https on {doc_id}")
    if record["modality"] not in VALID_MODALITY:
        raise ValueError(f"{folder}: bad modality on {doc_id}")
    if not content(record):
        raise ValueError(f"{folder}: empty content on {doc_id}")
    if not isinstance(record["metadata"], dict):
        raise ValueError(f"{folder}: metadata must be an object on {doc_id}")
    if datetime.fromisoformat(record["retrieved_at"].replace("Z", "+00:00")).tzinfo is None:
        raise ValueError(f"{folder}: retrieved_at must be timezone-aware on {doc_id}")


def read_sources(subtopic: dict, path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [{"subtopic": subtopic["folder"], **row} for row in csv.DictReader(f)]


def read_labels(subtopic: dict, path: Path, corpus_ids: set[str]) -> list[dict]:
    labels = []
    for row in load_jsonl(path):
        if row["doc_id"] not in corpus_ids:
            raise ValueError(f"{subtopic['folder']}: label for unknown doc_id {row['doc_id']}")
        labels.append({
            "doc_id": row["doc_id"],
            "member": subtopic["member"],
            "subtopic_key": subtopic["key"],
            "human_label": row["human_label"],
        })
    return labels


def read_taxonomy(subtopic: dict, folder_dir: Path) -> dict:
    path = folder_dir / "taxonomy.json"
    entry = {"member": subtopic["member"], "title": subtopic["title"]}
    if path.exists():
        taxonomy = json.loads(path.read_text(encoding="utf-8"))
        entry["fields"] = taxonomy.get("fields", {})
    else:
        entry["fields"] = None
        entry["defined_in"] = f"corpora/{subtopic['folder']}/extraction.py"
    return entry


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    all_rows: list[dict] = []
    all_labels: list[dict] = []
    all_sources: list[dict] = []
    taxonomies: dict[str, dict] = {}
    seen_ids: set[str] = set()
    stats: dict[str, dict] = {}

    for subtopic in SUBTOPICS:
        folder = subtopic["folder"]
        corpus_path = CORPORA_DIR / folder / "corpus.jsonl"
        if not corpus_path.exists():
            print(f"ERROR: missing {corpus_path}", file=sys.stderr)
            return 1

        rows = load_jsonl(corpus_path)
        redacted = 0
        for record in rows:
            validate_record(record, folder)
            if record["doc_id"] in seen_ids:
                raise ValueError(f"Duplicate doc_id across corpora: {record['doc_id']}")
            seen_ids.add(record["doc_id"])

            stamped = deepcopy(record)
            removed: set[str] = set()
            stamped["table_json"] = redact(stamped["table_json"], removed)
            meta = stamped["metadata"]
            meta["member"] = subtopic["member"]
            meta["subtopic_key"] = subtopic["key"]
            meta.setdefault("subtopic", subtopic["title"])
            if removed:
                meta["redacted_fields"] = sorted(removed)
                redacted += 1
            all_rows.append(stamped)

        folder_dir = CORPORA_DIR / folder
        labels = read_labels(subtopic, folder_dir / "human_labels.jsonl", {r["doc_id"] for r in rows})
        all_labels.extend(labels)
        taxonomies[folder] = read_taxonomy(subtopic, folder_dir)

        modality = Counter(r["modality"] for r in rows)
        tabular = sum(1 for r in rows if r["table_json"] is not None)
        stats[folder] = {
            "lead": subtopic["lead"],
            "title": subtopic["title"],
            "member": subtopic["member"],
            "records": len(rows),
            "human_labels": len(labels),
            "modality": dict(sorted(modality.items())),
            "tabular_share": round(tabular / len(rows), 3),
            "records_with_location_fields_removed": redacted,
        }
        all_sources.extend(read_sources(subtopic, folder_dir / "sources.csv"))

    out_corpus = ALL_DIR / "corpus.jsonl"
    out_sources = ALL_DIR / "sources.csv"
    out_stats = ALL_DIR / "corpus_stats.json"

    write_jsonl(out_corpus, all_rows)
    write_jsonl(ALL_DIR / "human_labels.jsonl", all_labels)
    (ALL_DIR / "taxonomy.json").write_text(json.dumps({
        "group_topic": "Pittsburgh 311 Municipal Service Resolution",
        "category_groups": CATEGORY_GROUPS,
        "subtopics": taxonomies,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fieldnames: list[str] = []
    for row in all_sources:
        fieldnames += [k for k in row if k not in fieldnames]
    with out_sources.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or ["subtopic"])
        writer.writeheader()
        writer.writerows(all_sources)

    tabular_total = sum(1 for r in all_rows if r["table_json"] is not None)
    out_stats.write_text(json.dumps({
        "records": len(all_rows),
        "human_labels": len(all_labels),
        "tabular_share": round(tabular_total / len(all_rows), 3),
        "subtopics": stats,
    }, indent=2) + "\n", encoding="utf-8")

    shutil.copy2(out_corpus, APPENDIX_ORIGINAL)

    for folder, s in stats.items():
        print(f"  {folder}: {s['records']} records, tabular {s['tabular_share']:.0%}, "
              f"location fields removed on {s['records_with_location_fields_removed']}")
    print(f"  total: {len(all_rows)} records, {len(all_labels)} human labels "
          f"-> {ALL_DIR.relative_to(REPO_ROOT)}/")
    print(f"  synced {APPENDIX_ORIGINAL.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
