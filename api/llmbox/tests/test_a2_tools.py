"""[A2 M3] tools, allowlist/argument validation and the multi-round loop (no model)."""
import json
from pathlib import Path
from types import SimpleNamespace

from src import tools
from src.generation import GenerationManager
from src.tools import check_call

SPEC = json.load(open("conf/tools/waste311_tools.json"))
OFFERED = {t["name"]: t for t in SPEC["tools"]}


def _data_dir():
    # course repo layout on the Mac, or a2/data/a1 next to llmbox on the VM
    for d in [Path("../../assignment01/assignment1/support/data"), Path("../a2/data/a1")]:
        if (d / "311_issue_category_codebook.csv").exists():
            return d
    raise FileNotFoundError("311 codebook CSV not found")


def setup_module(_):
    d = _data_dir()
    tc = SimpleNamespace(codebook_path=str(d / "311_issue_category_codebook.csv"),
                         issue_summary_path=str(d / "waste_issue_summary.csv"),
                         review_queue_path=None)
    tools.configure_waste311(SimpleNamespace(tool_calling=tc))


def test_allowlist_blocks_registered_but_unoffered_tool():
    assert "calculate" in tools.TOOL_REGISTRY
    assert "not offered" in check_call({"name": "calculate", "arguments": {"expression": "1+1"}}, OFFERED)


def test_argument_checks():
    assert "missing" in check_call({"name": "get_issue_stats", "arguments": {}}, OFFERED)
    assert "unknown" in check_call({"name": "get_issue_stats", "arguments": {"issue": "x", "doc_id": "y"}}, OFFERED)
    assert "must be one of" in check_call({"name": "queue_tag_update", "arguments": {
        "service_focus": "trash", "information_type": "service_guidance", "responsible_department": []}}, OFFERED)
    assert check_call({"name": "lookup_codebook", "arguments": {"query": "graffiti"}}, OFFERED) is None


def test_lookup_and_stats():
    m = tools.lookup_codebook("graffiti removal")["matches"]
    assert m and any("Graffiti" in x["issue"] or x["category"] == "Graffiti Issues" for x in m)
    s = tools.get_issue_stats("Early Set Out")
    assert s["issue"] == "Early Set Out" and s["median_hours"] > 0


def test_queue_writes_pending_review_for_runner_doc(tmp_path):
    tools.CONTEXT["review_queue"] = tmp_path / "q.jsonl"
    tools.CONTEXT["doc_id"] = "waste_x"
    assert tools.queue_tag_update("litter", "service_guidance", ["DPW"])["status"] == "pending_human_review"
    entry = json.loads((tmp_path / "q.jsonl").read_text())
    assert entry["doc_id"] == "waste_x" and entry["status"] == "pending_human_review"
    tools.CONTEXT["review_queue"] = None


def _fake(replies):
    replies = list(replies)
    return lambda messages: {"text": replies.pop(0), "prompt_tokens": 5, "completion_tokens": 2,
                             "latency_s": 0.1, "finish_reason": "stop", "seed": 1}


def test_loop_rejects_then_answers_and_counts_rounds():
    gm = GenerationManager()
    replies = ['functools[{"name": "calculate", "arguments": {"expression": "2*2"}}]',
               'functools[{"name": "lookup_codebook", "arguments": {"query": "litter can"}}]',
               '{"service_focus": "litter"}']
    out = gm.run_tool_loop(_fake(replies), [], OFFERED, max_rounds=3)
    assert out["tool_rounds"] == 2 and out["text"].startswith("{")
    assert out["tool_calls"][0]["rejected"] and out["tool_calls"][1]["rejected"] is None
    assert out["prompt_tokens"] == 15


def test_loop_stops_at_round_limit():
    gm = GenerationManager()
    call = 'functools[{"name": "lookup_codebook", "arguments": {"query": "x y"}}]'
    out = gm.run_tool_loop(_fake([call] * 5), [], OFFERED, max_rounds=2)
    assert out["tool_rounds"] == 2 and out["hit_round_limit"]


def test_required_choice_changes_prompt_and_nudges_once():
    gm = GenerationManager()
    assert "internal knowledge" in gm.build_functools_system_prompt("", [], "auto")
    req = gm.build_functools_system_prompt("", [], "required")
    assert "must call" in req and "internal knowledge" not in req
    out = gm.run_tool_loop(_fake(['{"a": 1}', 'functools[{"name": "lookup_codebook", "arguments": {"query": "litter"}}]',
                                  '{"a": 2}']), [], OFFERED, max_rounds=3, require_call=True)
    assert out["nudged"] and out["tool_rounds"] == 1 and out["text"] == '{"a": 2}'
    out = gm.run_tool_loop(_fake(['{"a": 1}', '{"a": 1}']), [], OFFERED, require_call=True)
    assert out["nudged"] and out["tool_rounds"] == 0
