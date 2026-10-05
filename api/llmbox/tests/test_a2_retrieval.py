"""[A2 M5] retrieval few-shot: ranking, self-exclusion, message shape."""
import json

from src.batch import compose_user_message
from src.retrieval import FewShotPool, TfidfIndex


def test_tfidf_ranks_overlap_first_and_excludes():
    idx = TfidfIndex([("a", "graffiti removal on walls"), ("b", "bulk trash pickup day"), ("c", "graffiti paint")])
    assert idx.search("graffiti removal on walls downtown", 2)[0][0] == "a"
    assert "a" not in [i for i, _ in idx.search("graffiti removal", 3, exclude={"a"})]


def test_pool_builds_alternating_turns_and_never_self(tmp_path):
    p = tmp_path / "pool.jsonl"
    rows = [{"id": "d1", "record": "graffiti removal request", "label": {"service_focus": "graffiti"}},
            {"id": "d2", "record": "recycling collection schedule", "label": {"service_focus": "collection"}}]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    pool = FewShotPool(p)
    msgs, got = pool.shots({"id": "d1", "instruction": "Tag.", "record": "graffiti on a bridge"}, 2, compose_user_message)
    assert [g["id"] for g in got] == ["d2"]
    assert [m["role"] for m in msgs] == ["user", "assistant"] and json.loads(msgs[1]["content"])["service_focus"] == "collection"
