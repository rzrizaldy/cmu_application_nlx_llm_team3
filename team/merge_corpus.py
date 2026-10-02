#!/usr/bin/env python3
"""Validate member corpora and write the combined team corpus + sources."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from category_groups import MEMBER_DISPLAY, MEMBER_SUBTOPIC

TEAM_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEAM_DIR.parent
MEMBERS_DIR = REPO_ROOT / "members"
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


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def content(record: dict) -> str:
    text = record.get("raw_text") or ""
    table = record.get("table_json")
    if table is not None:
        text += "\nTable:\n" + json.dumps(table, ensure_ascii=False, sort_keys=True)
    return text.strip()


def validate_record(record: dict, member: str) -> None:
    missing = REQUIRED_TOP - set(record.keys())
    extra = set(record.keys()) - REQUIRED_TOP
    if missing:
        raise ValueError(f"{member}: missing fields {missing} on {record.get('doc_id')}")
    if extra:
        raise ValueError(f"{member}: unexpected fields {extra} on {record.get('doc_id')}")
    if not record["source_url"].startswith("https://"):
        raise ValueError(f"{member}: source_url must be https for {record['doc_id']}")
    if record["modality"] not in VALID_MODALITY:
        raise ValueError(f"{member}: bad modality on {record['doc_id']}")
    if not content(record):
        raise ValueError(f"{member}: empty content on {record['doc_id']}")
    if not isinstance(record["metadata"], dict):
        raise ValueError(f"{member}: metadata must be object on {record['doc_id']}")
    retrieved = record["retrieved_at"].replace("Z", "+00:00")
    if datetime.fromisoformat(retrieved).tzinfo is None:
        raise ValueError(f"{member}: retrieved_at must be timezone-aware on {record['doc_id']}")


def merge_sources(member: str, path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            out = dict(row)
            out["member"] = member
            rows.append(out)
        return rows


def main() -> int:
    all_rows: list[dict] = []
    all_sources: list[dict] = []
    seen_ids: set[str] = set()
    counts: dict[str, int] = {}

    for member in sorted(MEMBER_SUBTOPIC.keys()):
        member_dir = MEMBERS_DIR / member
        corpus_path = member_dir / "corpus.jsonl"
        if not corpus_path.exists():
            print(f"ERROR: missing {corpus_path}", file=sys.stderr)
            return 1

        member_rows = load_jsonl(corpus_path)
        subtopic_key = MEMBER_SUBTOPIC[member]
        counts[member] = len(member_rows)

        for record in member_rows:
            validate_record(record, member)
            doc_id = record["doc_id"]
            if doc_id in seen_ids:
                raise ValueError(f"Duplicate doc_id across team: {doc_id}")
            seen_ids.add(doc_id)

            stamped = deepcopy(record)
            meta = stamped.setdefault("metadata", {})
            meta["member"] = member
            meta["subtopic_key"] = subtopic_key
            if "subtopic" not in meta:
                meta["subtopic"] = MEMBER_DISPLAY[member]
            all_rows.append(stamped)

        all_sources.extend(merge_sources(member, member_dir / "sources.csv"))

    out_corpus = TEAM_DIR / "corpus.jsonl"
    out_sources = TEAM_DIR / "sources.csv"

    with out_corpus.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if all_sources:
        fieldnames: list[str] = []
        for row in all_sources:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with out_sources.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_sources)
    else:
        out_sources.write_text("member,note\n", encoding="utf-8")

    print("Team corpus merge complete.")
    for member, n in counts.items():
        print(f"  {member}: {n} records ({MEMBER_SUBTOPIC[member]})")
    print(f"  total: {len(all_rows)} records")
    print(f"  wrote {out_corpus}")
    print(f"  wrote {out_sources}")

    shutil.copy2(out_corpus, APPENDIX_ORIGINAL)
    print(f"  synced {APPENDIX_ORIGINAL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
