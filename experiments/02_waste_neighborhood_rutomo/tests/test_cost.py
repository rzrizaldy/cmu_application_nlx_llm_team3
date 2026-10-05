import json
import sys
from pathlib import Path

from openpyxl import Workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cost  # noqa: E402


def test_cost_uses_observed_usable_rate_for_reviewed_run(tmp_path, monkeypatch):
    labels, runs, results, data = (tmp_path / x for x in ("labels", "runs", "results", "data"))
    for path in (labels, runs, results, data):
        path.mkdir()
    monkeypatch.setattr(cost, "LABELS", labels)
    monkeypatch.setattr(cost, "RUNS", runs)
    monkeypatch.setattr(cost, "RESULTS", results)
    monkeypatch.setattr(cost, "ASSUMPTIONS", data / "cost_assumptions.json")
    cost.ASSUMPTIONS.write_text(json.dumps({"staff_cost_per_hour_usd": 40, "staff_cost_source": "fixture",
        "compute_cost_per_hour_usd": 1, "compute_cost_source": "fixture", "setup_hours": 1,
        "setup_hours_source": "fixture", "annual_record_volume": 100, "annual_volume_source": "fixture",
        "manual_seconds_per_record": 60, "manual_seconds_source": "fixture"}))
    wb = Workbook()
    ws = wb.active
    ws.title = "Review"
    ws.append(["row", "doc_id", "record", "api_output", "usable_as_is (yes/no)", "notes"])
    ws.append([1, "a", "record", "{}", "yes", ""])
    ws.append([2, "b", "record", "{}", "no", ""])
    timing = wb.create_sheet("Timing")
    timing.append(["run", "rows", "start", "end", "date"])
    timing.append(["reviewed", 2, "10:00", "10:02", "2026-09-25"])
    wb.save(labels / "review_a2.xlsx")
    for name in ("reviewed", "other"):
        d = runs / name
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps({"exact_match": .75, "latency_s": {"mean": 2},
            "tokens": {"total_per_request": 100}}))
    cost.model(["reviewed", "other"])
    out = json.loads((results / "cost_model.json").read_text())["runs"]
    assert out["reviewed"]["usable_rate"] == .5
    assert out["reviewed"]["usable_rate_source"] == "timed_human_review"
    assert out["other"]["usable_rate"] == .75
    assert out["other"]["usable_rate_source"] == "exact_match_proxy"


def test_cost_requires_sourced_manual_time(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(cost, "ASSUMPTIONS", tmp_path / "cost_assumptions.json")
    base = {k: 1 for k in ["staff_cost_per_hour_usd", "compute_cost_per_hour_usd", "setup_hours", "annual_record_volume"]}
    base.update({k: "fixture" for k in ["staff_cost_source", "compute_cost_source", "setup_hours_source", "annual_volume_source"]})
    cost.ASSUMPTIONS.write_text(json.dumps({**base, "manual_seconds_per_record": 60, "manual_seconds_source": ""}))
    with pytest.raises(SystemExit, match="manual_seconds_source"):
        cost.model([])
