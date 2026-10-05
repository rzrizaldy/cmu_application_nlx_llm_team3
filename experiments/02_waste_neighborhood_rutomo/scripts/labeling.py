"""Blind human labelling for the 100 A2 reference records (DEV-50 + EVAL-50).

  python labeling.py export   -> labels/labeling_a2.xlsx   (blank; no model output, no prior labels)
  python labeling.py import   -> labels/human_labels_a2.jsonl + labels/labeling_timing.json

The workbook has three sheets:
  Labels  one row per record in a seeded random order (the split is not shown).
          Dropdowns for service_focus and information_type. responsible_department
          is exact names copied from the content, separated by ';', or NONE.
  Guide   the label definitions (same as A1).
  Timing  write the clock time when you start and finish each block of rows.
          This gives seconds per record, the manual-process baseline for Part E.

Import applies the A1 rules: valid enums, and every department must appear
verbatim in the record content.
"""
import json
import random
import sys
from pathlib import Path
from datetime import datetime

import jsonschema

from a2common import (BASELINE, FOCUS, KINDS, LABELS, SCHEMA, content, corpus, mask, save_json, save_jsonl,
                      split_manifest)

XLSX = LABELS / "labeling_a2.xlsx"
COLUMNS = ["row", "doc_id", "document_content", "service_focus", "information_type", "responsible_department", "notes"]


def excel_safe(text):
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) else text


def target_ids():
    split = split_manifest()
    ids = sorted(split["dev50"] + split["eval50"])
    random.Random(820952).shuffle(ids)
    return ids


def export():
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation

    if XLSX.exists():
        raise SystemExit(f"{XLSX} already exists; not overwriting your labels.")
    records = {r["doc_id"]: r for r in corpus()}
    wb = Workbook()
    ws = wb.active
    ws.title = "Labels"
    ws.append(COLUMNS)
    for c in ws[1]:
        c.font = Font(bold=True)
    for i, doc_id in enumerate(target_ids(), 1):
        text, _ = mask(content(records[doc_id]))
        ws.append([i, doc_id, excel_safe(text), "", "", "", ""])
    n = ws.max_row
    for col, values in [("D", FOCUS), ("E", KINDS)]:
        dv = DataValidation(type="list", formula1='"' + ",".join(values) + '"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"{col}2:{col}{n}")
    for col, width in zip("ABCDEFG", [5, 26, 90, 20, 22, 40, 30]):
        ws.column_dimensions[col].width = width
    for row in ws.iter_rows(min_row=2, min_col=3, max_col=3):
        row[0].alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "D2"

    guide = wb.create_sheet("Guide")
    for line in BASELINE.splitlines():
        guide.append([line])
    guide.append([""])
    guide.append(["responsible_department: copy exact names from document_content, separate with ';'. Write NONE if none is stated."])
    guide.column_dimensions["A"].width = 140

    timing = wb.create_sheet("Timing")
    timing.append(["block", "first_row", "last_row", "start_time (HH:MM)", "end_time (HH:MM)", "date (YYYY-MM-DD)"])
    for b, start in enumerate(range(1, 101, 10), 1):
        timing.append([b, start, start + 9, "", "", ""])
    XLSX.parent.mkdir(parents=True, exist_ok=True)
    wb.save(XLSX)
    print(f"wrote {XLSX} ({n - 1} records)")


def _cell(v):
    return "" if v is None else str(v).strip()


def _validate_and_save(entries, source):
    """entries: dicts {where, doc_id, label, notes, extra}. Applies the A1 label rules
    (valid enums; every department copied verbatim from the record), then writes
    human_labels_a2.jsonl. Both the workbook and the lab export go through here."""
    records = {r["doc_id"]: r for r in corpus()}
    expected = set(target_ids())
    labels, errors, seen = [], [], set()
    for e in entries:
        seen.add(e["doc_id"])
        try:
            if e["doc_id"] not in expected:
                raise ValueError("not one of the 100 A2 records")
            jsonschema.validate(e["label"], SCHEMA)
            missing = [d for d in e["label"]["responsible_department"] if d not in content(records[e["doc_id"]])]
            if missing:
                raise ValueError(f"department not found verbatim in the content: {missing}")
            labels.append({"doc_id": e["doc_id"], "human_label": e["label"], "notes": e.get("notes", ""),
                           "source": source, **e.get("extra", {})})
        except (ValueError, jsonschema.ValidationError) as err:
            errors.append(f"{e['where']} / {e['doc_id']}: {getattr(err, 'message', err)}")
    if seen != expected:
        errors.append(f"record set changed: missing {sorted(expected - seen)[:5]}, extra {sorted(seen - expected)[:5]}")
    if errors:
        raise SystemExit("Not imported. Fix these rows first:\n" + "\n".join(errors))
    save_jsonl(LABELS / "human_labels_a2.jsonl", sorted(labels, key=lambda r: r["doc_id"]))
    return labels


def import_():
    """Workbook (labels/labeling_a2.xlsx) -> human_labels_a2.jsonl + labeling_timing.json."""
    from openpyxl import load_workbook

    wb = load_workbook(XLSX, data_only=True)
    rows = list(wb["Labels"].values)
    if list(rows[0]) != COLUMNS:
        raise SystemExit("Labels header changed; keep the original columns.")
    entries = []
    for values in rows[1:]:
        row = dict(zip(COLUMNS, values))
        if not _cell(row["doc_id"]):
            continue
        dept = _cell(row["responsible_department"])
        if not dept:
            raise SystemExit(f"row {row['row']}: responsible_department is empty; write NONE if no department is stated")
        entries.append({"where": f"row {row['row']}", "doc_id": _cell(row["doc_id"]), "notes": _cell(row["notes"]),
                        "label": {"service_focus": _cell(row["service_focus"]),
                                  "information_type": _cell(row["information_type"]),
                                  "responsible_department": [] if dept.upper() == "NONE" else
                                  [d.strip() for d in dept.split(";") if d.strip()]}})
    labels = _validate_and_save(entries, "workbook")

    blocks = []
    for values in list(wb["Timing"].values)[1:]:
        block, first, last, start, end, date = values
        if start and end and date:
            fmt = "%Y-%m-%d %H:%M"
            s = datetime.strptime(f"{_cell(date)[:10]} {_cell(start)[:5]}", fmt)
            e = datetime.strptime(f"{_cell(date)[:10]} {_cell(end)[:5]}", fmt)
            blocks.append({"block": block, "rows": int(last) - int(first) + 1, "minutes": (e - s).total_seconds() / 60})
    total_rows = sum(b["rows"] for b in blocks)
    save_json(LABELS / "labeling_timing.json", {
        "method": "workbook Timing sheet (clock time per block of 10 rows)",
        "blocks": blocks,
        "timed_rows": total_rows,
        "seconds_per_record": round(sum(b["minutes"] for b in blocks) * 60 / total_rows, 1) if total_rows else None,
    })
    print(f"imported {len(labels)} labels; timed rows: {total_rows}")


def import_lab(path):
    """Label-studio export from the 311 NLX Lab (JSON, schema a2-labels/1) ->
    human_labels_a2.jsonl + labeling_timing.json. The lab times each record
    itself (active seconds, idle gaps over 3 minutes are not counted)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != "a2-labels/1":
        raise SystemExit(f"{path}: not a label-studio export (schema a2-labels/1)")
    entries = [{"where": f"lab #{i}", "doc_id": r["doc_id"], "notes": r.get("notes", ""),
                "label": {k: r[k] for k in ("service_focus", "information_type", "responsible_department")}}
               for i, r in enumerate(data["labels"], 1)]
    labels = _validate_and_save(entries, "student")
    timing_path = LABELS / "labeling_timing.json"
    if timing_path.exists():
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        if timing.get("seconds_per_record"):
            print(f"imported {len(labels)} labels from the lab; kept existing labeling_timing.json")
            return
    save_json(timing_path, {
        "method": "Fill in student-measured labeling time (workbook Timing sheet or total minutes / 100)",
        "timed_rows": 0,
        "seconds_per_record": None,
    })
    print(f"imported {len(labels)} labels from the lab; add labeling_timing.json before Part E cost modeling")


if __name__ == "__main__":
    {"export": export, "import": import_, "import-lab": lambda: import_lab(sys.argv[2])}[sys.argv[1]]()
