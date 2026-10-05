import json
import random
import csv

input_file = "outputs/baseline_eval1_responses.jsonl"
output_file = "outputs/baseline_eval1_manual_review.csv"

records = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip() != "":
            records.append(json.loads(line))

random.seed(42)
sample = random.sample(records, 10)

rows = []

for record in sample:
    response = record["model_response"]

    pred_request_type_id = ""
    pred_issue = ""
    pred_category = ""
    pred_department = ""

    for line in response.split("\n"):
        line = line.strip()

        if line.lower().startswith("request_type_id:"):
            pred_request_type_id = line.split(":", 1)[1].strip()

        elif line.lower().startswith("issue:"):
            pred_issue = line.split(":", 1)[1].strip()

        elif line.lower().startswith("category:"):
            pred_category = line.split(":", 1)[1].strip()

        elif line.lower().startswith("department:"):
            pred_department = line.split(":", 1)[1].strip()

    truth = record["ground_truth"]

    row = {
        "doc_id": record["doc_id"],
        "input": record["input"],
        "true_request_type_id": truth["request_type_id"],
        "pred_request_type_id": pred_request_type_id,
        "true_issue": truth["issue"],
        "pred_issue": pred_issue,
        "true_category": truth["category"],
        "pred_category": pred_category,
        "true_department": truth["department"],
        "pred_department": pred_department,
        "semantic_rating": "",
        "notes": ""
    }

    rows.append(row)

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Saved:", output_file)
print("Rows:", len(rows))