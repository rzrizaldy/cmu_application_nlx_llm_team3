"""[A2 M2] structured output: validation and retry loop, with a fake generator."""
from src.pydantic_models.waste_tags import WasteTags, WasteTagsDeptRequired
from src.structured import first_json_object, load_pydantic_model, schema_instruction, structured_generate

GOOD = '{"service_focus": "graffiti", "information_type": "service_guidance", "responsible_department": []}'


def fake(replies):
    replies = list(replies)
    seen = []

    def gen(messages):
        seen.append(list(messages))
        return {"text": replies.pop(0), "prompt_tokens": 10, "completion_tokens": 5, "latency_s": 0.5,
                "finish_reason": "stop", "seed": 1}
    return gen, seen


def test_valid_first_try_with_fence():
    gen, _ = fake(["```json\n" + GOOD + "\n```"])
    out = structured_generate(gen, [{"role": "user", "content": "x"}], WasteTags)
    assert out["retries"] == 0 and out["parsed"]["service_focus"] == "graffiti" and out["violation"] is None


def test_retry_feeds_error_back_and_sums_cost():
    bad = GOOD.replace("graffiti", "graffitti")
    gen, seen = fake([bad, GOOD])
    out = structured_generate(gen, [{"role": "user", "content": "x"}], WasteTags, max_retries=2)
    assert out["retries"] == 1 and out["parsed"] is not None
    assert out["prompt_tokens"] == 20 and out["completion_tokens"] == 10 and out["latency_s"] == 1.0
    assert "did not match" in seen[1][-1]["content"]


def test_gives_up_after_max_retries():
    gen, _ = fake(["nope", "still nope", "no"])
    out = structured_generate(gen, [{"role": "user", "content": "x"}], WasteTags, max_retries=2)
    assert out["parsed"] is None and out["retries"] == 2 and len(out["attempts"]) == 3


def test_dept_required_rejects_empty_list():
    gen, _ = fake([GOOD, GOOD])
    out = structured_generate(gen, [{"role": "user", "content": "x"}], WasteTagsDeptRequired, max_retries=1)
    assert out["parsed"] is None


def test_extra_keys_and_duplicates_rejected():
    for text in [GOOD[:-1] + ', "confidence": 0.9}',
                 GOOD.replace("[]", '["DPW", "DPW"]')]:
        gen, _ = fake([text])
        assert structured_generate(gen, [], WasteTags, max_retries=0)["parsed"] is None


def test_loader_and_schema_text():
    assert load_pydantic_model("src.pydantic_models.waste_tags:WasteTags") is WasteTags
    assert "weeds/debris" in schema_instruction(WasteTags)
    assert first_json_object('pre {"a": {"b": 1}} post') == '{"a": {"b": 1}}'
