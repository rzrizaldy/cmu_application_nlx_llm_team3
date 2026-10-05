import json

from safety_guardrail import guarded_route


adversarial_file = "data/pittsburgh311/adversarial_probes.jsonl"
benign_file = "data/pittsburgh311/benign_safety_inputs.jsonl"

output_file = "outputs/guardrail_eval_responses.jsonl"


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() != "":
                records.append(json.loads(line))

    return records


adversarial_records = load_jsonl(adversarial_file)
benign_records = load_jsonl(benign_file)


all_records = []

for record in adversarial_records:
    record["set_type"] = "adversarial"
    all_records.append(record)

for record in benign_records:
    record["set_type"] = "benign"
    record["category"] = "benign"
    all_records.append(record)


with open(output_file, "w", encoding="utf-8") as out_file:

    for i, record in enumerate(all_records, start=1):

        result = guarded_route(record["input"])

        output = {
            "id": record["id"],
            "set_type": record["set_type"],
            "category": record["category"],
            "input": record["input"],
            "blocked": result["blocked"],
            "reason": result["reason"],
            "response": result["response"],
            "latency_seconds": result["latency_seconds"]
        }

        out_file.write(
            json.dumps(output, ensure_ascii=False) + "\n"
        )

        print(
            f"[{i}/{len(all_records)}] "
            f"{record['id']} "
            f"blocked={result['blocked']} "
            f"reason={result['reason']}"
        )


print()
print("Finished.")
print("Saved responses to:", output_file)