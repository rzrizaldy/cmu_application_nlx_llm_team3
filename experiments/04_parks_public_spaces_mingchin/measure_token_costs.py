import json

from transformers import AutoTokenizer


model_path = "./models/llms/microsoft/phi-4-mini-instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def count_tokens(text):
    if text is None:
        return 0

    return len(
        tokenizer.encode(
            str(text),
            add_special_tokens=False
        )
    )


def get_response(record):
    if record.get("model_response"):
        return record["model_response"]

    if record.get("response"):
        return record["response"]

    return ""


def evaluate_file(name, path):
    records = load_jsonl(path)

    input_tokens = []
    output_tokens = []

    for record in records:
        user_input = record.get("input", "")
        response = get_response(record)

        input_tokens.append(
            count_tokens(user_input)
        )

        output_tokens.append(
            count_tokens(response)
        )

    avg_input = sum(input_tokens) / len(input_tokens)
    avg_output = sum(output_tokens) / len(output_tokens)
    avg_total = avg_input + avg_output

    print(name)
    print(f"  Records: {len(records)}")
    print(f"  Average input tokens: {avg_input:.1f}")
    print(f"  Average output tokens: {avg_output:.1f}")
    print(f"  Average total tokens: {avg_total:.1f}")
    print()


print("LLM Token Cost Summary")
print("----------------------")
print()


evaluate_file(
    "Baseline Eval 1",
    "outputs/baseline_eval1_responses.jsonl"
)

evaluate_file(
    "Baseline Eval 3",
    "outputs/baseline_eval3_responses.jsonl"
)

evaluate_file(
    "Structured Output",
    "outputs/structured_output_eval1_responses.jsonl"
)

evaluate_file(
    "Tool Calling",
    "outputs/tool_calling_eval2_responses.jsonl"
)


print("TF-IDF Top 1")
print("  Average LLM tokens: 0")
print()

print("TF-IDF Majority Vote")
print("  Average LLM tokens: 0")
print()

print("Guardrail")
print("  Added LLM tokens per request: 0")