import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from a2common import content, mask  # noqa: E402
from score import bootstrap_ci, macro_f1, parse_json  # noqa: E402


def test_parse_json_handles_fences_and_chatter():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": {"b": 2}} hope this helps') == {"a": {"b": 2}}
    assert parse_json("no json here") is None
    assert parse_json('{broken {"ok": true}') == {"ok": True}


def test_mask_keeps_floats_and_masks_pii_fin():
    assert mask("median 294.1333333333333 hours")[1] == {}
    assert mask("card 4111 1111 1111 1111")[0] == "card <FIN>"
    assert mask("call 412-255-2621 or a@b.org")[1] == {"PII": 2}
    assert mask("see 414 Grant Street today")[0] == "see <PII> today"


def test_content_excludes_metadata_label_leaks():
    r = {"raw_text": "hello", "table_json": {"x": 1}, "metadata": {"source_kind": "issue_taxonomy"},
         "source_url": "https://example.org/graffiti"}
    text = content(r)
    assert "issue_taxonomy" not in text and "graffiti" not in text and '"x": 1' in text


def test_macro_f1_and_ci():
    assert macro_f1(["a", "b"], ["a", "b"]) == 1.0
    assert macro_f1(["a", "a", "b"], ["a", "b", "b"]) == (2 / 3 + 2 / 3) / 2
    lo, hi = bootstrap_ci([1] * 30 + [0] * 20)
    assert lo < 0.6 < hi
