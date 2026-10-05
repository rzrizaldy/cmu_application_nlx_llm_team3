#!/usr/bin/env python3
"""Operational evidence for every issue in the four subtopics, from one WPRDC join.

Follows the brief's minimum cleaning and join contract: request_type_id read as
string, codebook rows with a blank category dropped, latest codebook_version kept
per request_type_id, then a many-to-one left join. Resolution time is
closed_date_utc - create_date_utc for closed requests, in days.

Writes operational_evidence.jsonl (corpus schema, one record per issue, no
location fields) and operational_evidence_summary.json (coverage counts,
including requests whose category is blank or unmapped).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from category_groups import CATEGORY_GROUPS

ALL_DIR = Path(__file__).resolve().parent
RAW_DIR = ALL_DIR.parent / "04_parks_public_spaces_mingchin"
REQUESTS_CSV = RAW_DIR / "311_data.csv"
CODEBOOK_CSV = RAW_DIR / "311_issue_category_codebook.csv"
SOURCE_URL = "https://data.wprdc.org/dataset/311-data"
LICENSE = "Creative Commons Attribution (CC-BY); City of Pittsburgh 311 Data Archive via WPRDC."
CATEGORY_TO_SUBTOPIC = {c: sk for sk, cats in CATEGORY_GROUPS.items() for c in cats}


def load_enriched() -> tuple[pd.DataFrame, pd.DataFrame]:
    requests = pd.read_csv(
        REQUESTS_CSV,
        dtype={"request_type_id": "string"},
        usecols=["request_type_id", "request_type_name", "status_name", "create_date_utc", "closed_date_utc"],
    )
    codebook = pd.read_csv(CODEBOOK_CSV, dtype={"request_type_id": "string"}, encoding="utf-8-sig")
    codebook_clean = (
        codebook.loc[codebook["request_type_id"].notna() & codebook["category"].fillna("").ne("")]
        .sort_values("codebook_version")
        .drop_duplicates("request_type_id", keep="last")
    )
    enriched = requests.merge(
        codebook_clean[["request_type_id", "issue", "category", "department", "definition"]],
        on="request_type_id", how="left", validate="many_to_one",
    )
    return enriched, codebook_clean


def resolution_days(df: pd.DataFrame) -> pd.Series:
    created = pd.to_datetime(df["create_date_utc"], errors="coerce", utc=True)
    closed = pd.to_datetime(df["closed_date_utc"], errors="coerce", utc=True)
    days = (closed - created).dt.total_seconds() / 86400
    return days[(df["status_name"].str.lower() == "closed") & days.notna() & (days >= 0)]


def fmt_days(d: float) -> str:
    return f"{d * 24:.1f} hours" if d < 1 else f"{d:.1f} days"


def main() -> int:
    enriched, codebook_clean = load_enriched()
    retrieved_at = datetime.fromtimestamp(REQUESTS_CSV.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    in_scope = enriched[enriched["category"].isin(CATEGORY_TO_SUBTOPIC)]

    records = []
    for (rtid, issue, category, department), grp in in_scope.groupby(
        ["request_type_id", "issue", "category", "department"], dropna=False, sort=True
    ):
        days = resolution_days(grp)
        stats = {
            "request_type_id": rtid,
            "issue": issue,
            "category": category,
            "department": department,
            "aliases": sorted({n.strip() for n in grp["request_type_name"].dropna()} - {issue.strip()}),
            "request_volume": int(len(grp)),
            "closed_requests": int(len(days)),
        }
        if len(days):
            q = days.quantile([0.5, 0.75, 0.9])
            stats |= {"median_days": round(q[0.5], 2), "p75_days": round(q[0.75], 2), "p90_days": round(q[0.9], 2)}
            timing = (
                f"Closed requests resolve in a median of {fmt_days(q[0.5])} "
                f"(75th percentile {fmt_days(q[0.75])}, 90th percentile {fmt_days(q[0.9])})."
            )
        else:
            timing = "No closed requests with valid dates."
        definition = grp["definition"].dropna().iloc[0] if grp["definition"].notna().any() else None
        text = (
            f"311 issue \"{issue}\" (request type {rtid}), category {category}, routed to {department}. "
            f"{len(grp)} requests in the archive, {len(days)} closed. {timing}"
        )
        if definition:
            text += f" Definition: {definition}"
        records.append({
            "doc_id": f"ops_{rtid}",
            "source_url": SOURCE_URL,
            "retrieved_at": retrieved_at,
            "modality": "mixed",
            "raw_text": text,
            "table_json": stats,
            "license_note": LICENSE,
            "metadata": {
                "group_topic": "Pittsburgh 311 municipal service requests",
                "subtopic_key": CATEGORY_TO_SUBTOPIC[category],
                "member": "team",
                "document_type": "resolution_statistics",
                "category": category,
                "department": department,
                "issue": issue,
                "request_type_id": rtid,
                "synthetic": False,
            },
        })

    out = ALL_DIR / "operational_evidence.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))

    blank = enriched["category"].isna()
    summary = {
        "requests_total": int(len(enriched)),
        "requests_in_four_subtopics": int(len(in_scope)),
        "requests_unmapped_blank_category": int(blank.sum()),
        "requests_mapped_other_categories": int((~blank & ~enriched["category"].isin(CATEGORY_TO_SUBTOPIC)).sum()),
        "codebook_rows_with_category": int(len(codebook_clean)),
        "issues_with_evidence": len(records),
        "by_subtopic": {
            sk: {
                "issues": sum(1 for r in records if r["metadata"]["subtopic_key"] == sk),
                "requests": int(in_scope["category"].isin(cats).sum()),
            }
            for sk, cats in CATEGORY_GROUPS.items()
        },
        "by_category": {c: int(n) for c, n in in_scope["category"].value_counts().sort_index().items()},
    }
    (ALL_DIR / "operational_evidence_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_category"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
