"""Label-studio import: same validation as the workbook, writes to a temp LABELS dir."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import labeling  # noqa: E402
from a2common import content, corpus, load_jsonl  # noqa: E402


def _export(tmp_path, mutate=None):
    ids = labeling.target_ids()
    text = {r["doc_id"]: content(r) for r in corpus()}
    rows = []
    for i, d in enumerate(ids):
        rows.append({"doc_id": d, "service_focus": "collection", "information_type": "service_guidance",
                     "responsible_department": [], "notes": ""})
    if mutate:
        mutate(rows, text)
    p = tmp_path / "export.json"
    p.write_text(json.dumps({"schema": "a2-labels/1", "exported_at": "2026-09-28T00:00:00Z", "labels": rows}))
    return p


def test_import_lab_writes_labels_and_timing(tmp_path, monkeypatch):
    monkeypatch.setattr(labeling, "LABELS", tmp_path)
    (tmp_path / "labeling_timing.json").write_text(json.dumps({"seconds_per_record": 18.0, "timed_rows": 100}))
    labeling.import_lab(_export(tmp_path))
    labels = load_jsonl(tmp_path / "human_labels_a2.jsonl")
    assert len(labels) == 100 and labels[0]["source"] == "student"
    timing = json.loads((tmp_path / "labeling_timing.json").read_text())
    assert timing["seconds_per_record"] == 18.0


def test_import_lab_rejects_invented_department(tmp_path, monkeypatch):
    monkeypatch.setattr(labeling, "LABELS", tmp_path)

    def bad(rows, text):
        rows[0]["responsible_department"] = ["Bureau of Labor Statistics"]
    with pytest.raises(SystemExit, match="not found verbatim"):
        labeling.import_lab(_export(tmp_path, bad))
    assert not (tmp_path / "human_labels_a2.jsonl").exists()


def test_import_lab_rejects_missing_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(labeling, "LABELS", tmp_path)
    with pytest.raises(SystemExit, match="record set changed"):
        labeling.import_lab(_export(tmp_path, lambda rows, text: rows.pop()))
