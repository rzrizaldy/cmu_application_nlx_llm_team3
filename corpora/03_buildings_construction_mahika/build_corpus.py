#!/usr/bin/env python3
"""Build corpus.jsonl (and sources.csv) for the Buildings, Construction &
Accessibility subtopic of the Pittsburgh 311 corpus (NL(X) & LLM Group 3).

    python build_corpus.py --input 311_archive.csv --codebook 311_codebook.csv --per-category 65

Every record is modality "mixed": raw_text is a natural-language description
composed from the request's fields, table_json holds the administrative fields
(covers the >=20% tabular requirement). The derived category is kept OUT of what
Phi sees (metadata only), so extracting issue_category stays a real task.

Column names are auto-detected case-insensitively and printed. Use
--list-types to see the request types, or --type-column / --category-column to
override detection.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from schema import validate_record

DEFAULT_SOURCE_URL = (
    "https://data.wprdc.org/dataset/311-data/resource/"
    "29462525-62a6-45bf-9b5e-ad2e1c06348d"
)
DEFAULT_SOURCE_NAME = "Pittsburgh 311 Data (archive), WPRDC"
DEFAULT_LICENSE = (
    "Creative Commons Attribution 4.0 (CC-BY); use under the WPRDC Data User Agreement."
)
DEFAULT_CATEGORIES = ["Building Maintenance", "Construction Issues", "Permits", "Accessibility"]

COLUMN_CANDIDATES: dict[str, list[str]] = {
    "request_id":     ["REQUEST_ID", "request_id", "_id", "id"],
    "group_id":       ["group_id"],
    "created_on":     ["CREATED_ON", "created_on", "create_date_et", "create_date_utc",
                       "create_date", "created_date", "created"],
    "closed_date":    ["closed_date_et", "closed_date_utc", "closed_date"],
    "last_action":    ["last_action_et", "last_action_utc"],
    "request_type":   ["request_type_name", "REQUEST_TYPE", "request_type", "Request Type",
                       "issue_type", "complaint_type", "issue", "type", "subject"],
    "request_type_id":["request_type_id"],
    "category":       ["CATEGORY", "category", "Category", "Issue Category", "issue_category"],
    "origin":         ["REQUEST_ORIGIN", "request_origin", "origin", "Origin", "source"],
    "status":         ["status_name", "STATUS", "status", "Status", "state"],
    "status_code":    ["status_code"],
    "department":     ["DEPARTMENT", "department", "dept", "Department", "division"],
    "neighborhood":   ["NEIGHBORHOOD", "neighborhood", "Neighborhood"],
    "street":         ["street"],
    "cross_street":   ["cross_street"],
    "city":           ["city"],
    "council":        ["COUNCIL_DISTRICT", "council_district"],
    "ward":           ["WARD", "ward"],
    "tract":          ["TRACT", "census_tract", "tract"],
    "police_zone":    ["POLICE_ZONE", "police_zone"],
    "x":              ["X", "longitude", "LONGITUDE", "lon"],
    "y":              ["Y", "latitude", "LATITUDE", "lat"],
    "geo_accuracy":   ["GEO_ACCURACY", "geo_accuracy"],
}

TABLE_KEYS = [
    "request_id", "group_id", "created_on", "closed_date", "request_type",
    "request_type_id", "origin", "status", "status_code", "department",
    "neighborhood", "street", "cross_street", "city", "council", "ward",
    "tract", "police_zone", "x", "y", "geo_accuracy",
]


def resolve_columns(header: list[str]) -> dict[str, str]:
    lower = {h.lower(): h for h in header}
    resolved = {}
    for logical, candidates in COLUMN_CANDIDATES.items():
        for cand in candidates:
            if cand.lower() in lower:
                resolved[logical] = lower[cand.lower()]
                break
    return resolved


def load_codebook(path: Path) -> dict[str, str]:
    """Build a lookup {request-type key (lowercased) -> category}. Keys from every
    plausible key column present (name and id), so the join works either way."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    lower = {h.lower(): h for h in rows[0].keys()}
    key_cols = [lower[c] for c in
                ["request_type_name", "request_type", "issue", "issue name",
                 "request type", "name", "request_type_id"] if c in lower]
    cat_col = next((lower[c] for c in
                    ["category", "issue category", "group", "request_type_category"]
                    if c in lower), None)
    print(f"Codebook columns: {list(rows[0].keys())}")
    if not key_cols or not cat_col:
        print("[warn] could not find a request-type key column and a category column in the "
              "codebook; ignoring it.", file=sys.stderr)
        return {}
    print(f"  joining on {key_cols} -> {cat_col!r}")
    mapping: dict[str, str] = {}
    for r in rows:
        cat = (r.get(cat_col) or "").strip()
        if not cat:
            continue
        for kc in key_cols:
            key = (r.get(kc) or "").strip().lower()
            if key:
                mapping[key] = cat
    return mapping


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def status_text(value: str) -> str:
    v = str(value).strip()
    if v in ("1", "1.0"):
        return "closed"
    if v in ("0", "0.0"):
        return "open"
    return v or "unknown"


def category_of(row, cols, codebook):
    if cols.get("category"):
        return (row.get(cols["category"]) or "").strip() or None
    if codebook:
        for key in ("request_type", "request_type_id"):
            col = cols.get(key)
            if col:
                cat = codebook.get((row.get(col) or "").strip().lower())
                if cat:
                    return cat
    return None


def describe(row, cols) -> str:
    def g(key):
        col = cols.get(key)
        return (row.get(col) or "").strip() if col else ""

    rtype = g("request_type") or "an unspecified issue"
    hood, street = g("neighborhood"), g("street")
    dept, origin, created = g("department"), g("origin"), g("created_on")
    status = status_text(g("status")) if cols.get("status") else ""

    where = f"in the {hood} neighborhood of Pittsburgh" if hood else "in the City of Pittsburgh"
    text = f'A resident submitted a 311 service request about "{rtype}" {where}.'
    if street:
        text += f" The reported location is near {street}."
    extras = []
    if origin:
        extras.append(f"submitted via {origin}")
    if created:
        extras.append(f"on {created}")
    if extras:
        text += " The request was " + " ".join(extras) + "."
    if dept:
        text += f" It was routed to {dept}."
    if status:
        text += f" Current status: {status}."
    return text


def make_table(row, cols) -> dict:
    table = {}
    for key in TABLE_KEYS:
        col = cols.get(key)
        if col:
            val = (row.get(col) or "").strip()
            if val:
                table[key] = val
    return table


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--codebook", default=None)
    parser.add_argument("--out", default="corpus.jsonl")
    parser.add_argument("--sources", default="sources.csv")
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--per-category", type=int, default=65)
    parser.add_argument("--min-total", type=int, default=150)
    parser.add_argument("--max-total", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--source-name", default=DEFAULT_SOURCE_NAME)
    parser.add_argument("--license-note", default=DEFAULT_LICENSE)
    parser.add_argument("--id-prefix", default="bca")
    parser.add_argument("--type-column", default=None)
    parser.add_argument("--category-column", default=None)
    parser.add_argument("--list-types", action="store_true",
                        help="print the most common request types and exit")
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"No such file: {src}")

    with open(src, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        rows = list(reader)
    cols = resolve_columns(header)
    if args.type_column:
        cols["request_type"] = args.type_column
    if args.category_column:
        cols["category"] = args.category_column

    print(f"Read {len(rows):,} rows from {src.name}")
    print("\nAll columns in this file:")
    for i, name in enumerate(header, 1):
        print(f"  {i:>2}. {name}")
    print(f"\nResolved columns: {cols}")

    if "request_type" not in cols:
        raise SystemExit(
            "\nCould not auto-detect the request-type/issue column from the names above.\n"
            "Rerun with --type-column \"<exact name>\".")

    if args.list_types:
        col = cols["request_type"]
        counts = Counter((r.get(col) or "").strip() for r in rows)
        print(f"\nTop request types in {col!r}:")
        for name, n in counts.most_common(60):
            print(f"  {n:>7,}  {name}")
        return

    codebook = load_codebook(Path(args.codebook)) if args.codebook else {}
    if "category" not in cols and not codebook:
        raise SystemExit(
            "\nThe data has no category column and no usable codebook was given.\n"
            "Download the '311 Issue and Category Codebook' (resource 6a2c9de6-...), save it as\n"
            "311_codebook.csv, and pass --codebook 311_codebook.csv")

    derived = [category_of(r, cols, codebook) for r in rows]
    all_cats = Counter(c for c in derived if c)
    print("\nTop categories found (after codebook join):")
    for cat, n in all_cats.most_common(25):
        print(f"  {n:>7,}  {cat}")
    if codebook and not all_cats:
        col = cols["request_type"]
        unmapped = Counter((r.get(col) or "").strip() for r, c in zip(rows, derived) if not c)
        print("\n[warn] nothing joined. The codebook key does not match the data's request types.")
        print("Most common UNMAPPED request types (share these with me):")
        for name, n in unmapped.most_common(15):
            print(f"  {n:>7,}  {name}")
        raise SystemExit(1)

    targets = {c.strip().lower(): c for c in args.categories}
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row, cat in zip(rows, derived):
        if cat and cat.strip().lower() in targets:
            buckets[targets[cat.strip().lower()]].append(row)

    print(f"\nMatched your {len(args.categories)} target categories:")
    for cat in args.categories:
        print(f"  {len(buckets.get(cat, [])):>7,}  {cat}")

    rng = random.Random(args.seed)
    chosen: list[tuple[str, dict]] = []
    for cat in args.categories:
        pool = buckets.get(cat, [])
        take = pool if len(pool) <= args.per_category else rng.sample(pool, args.per_category)
        chosen.extend((cat, row) for row in take)
    rng.shuffle(chosen)
    if len(chosen) > args.max_total:
        chosen = chosen[:args.max_total]

    id_col = cols.get("request_id", "")
    records, problems = [], []
    for i, (cat, row) in enumerate(sorted(chosen, key=lambda cr: cr[1].get(id_col, "")), start=1):
        record = {
            "doc_id": f"{args.id_prefix}_{i:04d}",
            "source_url": args.source_url,
            "retrieved_at": now_iso(),
            "modality": "mixed",
            "raw_text": describe(row, cols),
            "table_json": make_table(row, cols),
            "license_note": args.license_note,
            "metadata": {
                "group_topic": "pittsburgh 311 service requests",
                "subtopic": "buildings, construction, and accessibility",
                "document_type": "service_request",
                "request_id": (row.get(id_col) or "").strip() if id_col else None,
                "codebook_category": cat,
            },
        }
        records.append(record)
        problems.extend(validate_record(record, i))

    with open(args.out, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    tabular = sum(1 for r in records if r["modality"] in ("table", "mixed") and r["table_json"])
    with open(args.sources, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_url", "source_name", "retrieved_at", "license_note", "record_count"])
        writer.writerow([args.source_url, args.source_name, now_iso(),
                         args.license_note, len(records)])

    print(f"\nWrote {len(records)} records to {args.out}")
    print(f"  tabular share: {tabular}/{len(records)} ({tabular / max(len(records), 1):.0%})")
    print("  per category:")
    per = Counter(r["metadata"]["codebook_category"] for r in records)
    for cat in args.categories:
        print(f"    {per.get(cat, 0):>4}  {cat}")
    print(f"Wrote sources.csv -> {args.sources}")
    if problems:
        print(f"\n{len(problems)} schema problem(s):")
        for p in problems[:10]:
            print(f"  ! {p}")
    if len(records) < args.min_total:
        print(f"\n[warn] only {len(records)} records (< {args.min_total}). Raise --per-category.")
    else:
        print("\nNext: python check_corpus.py corpus.jsonl --sources sources.csv")


if __name__ == "__main__":
    main()
