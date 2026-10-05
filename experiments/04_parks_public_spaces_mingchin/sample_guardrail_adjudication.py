import csv
import json
from collections import defaultdict


input_file = "outputs/guardrail_eval_responses.jsonl"
output_file = "outputs/guardrail_human_review.csv"


records = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip() != "":
            records.append(json.loads(line))


adversarial_by_category = defaultdict(list)
benign_records = []

for record in records:
    if record["set_type"] == "adversarial":
        adversarial_by_category[record["category"]].append(record)
    else:
        benign_records.append(record)


sample = []

# Take 2 examples from each adversarial category
for category in adversarial_by_category:
    sample.extend(adversarial_by_category[category][:2])

# Take 4 benign examples
sample.extend(benign_records[:4])


with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "id",
        "category",
        "input",
        "guardrail_decision",
        "guardrail_reason",
        "human_correct",
        "notes"
    ])

    for record in sample:

        decision = "block" if record["blocked"] else "allow"

        writer.writerow([
            record["id"],
            record["category"],
            record["input"],
            decision,
            record["reason"],
            "",
            ""
        ])


print("Sample size:", len(sample))
print("Saved:", output_file)