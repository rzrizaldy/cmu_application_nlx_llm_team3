import json
import re


input_file = "outputs/tfidf_vote_llm_eval4_responses.jsonl"


records = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))


def clean(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def extract_field(text, field_name):
    pattern = rf"{field_name}\s*:\s*(.+)"
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        return match.group(1).strip()

    return ""


total = len(records)

request_type_correct = 0
issue_correct = 0
category_correct = 0
department_correct = 0
full_correct = 0

total_latency = 0


for record in records:
    response = record["model_response"]
    ground_truth = record["ground_truth"]

    prediction = {
        "request_type_id": extract_field(
            response,
            "request_type_id"
        ),
        "issue": extract_field(
            response,
            "issue"
        ),
        "category": extract_field(
            response,
            "category"
        ),
        "department": extract_field(
            response,
            "department"
        )
    }

    request_match = (
        clean(prediction["request_type_id"])
        == clean(ground_truth["request_type_id"])
    )

    issue_match = (
        clean(prediction["issue"])
        == clean(ground_truth["issue"])
    )

    category_match = (
        clean(prediction["category"])
        == clean(ground_truth["category"])
    )

    department_match = (
        clean(prediction["department"])
        == clean(ground_truth["department"])
    )

    if request_match:
        request_type_correct += 1

    if issue_match:
        issue_correct += 1

    if category_match:
        category_correct += 1

    if department_match:
        department_correct += 1

    if (
        request_match
        and issue_match
        and category_match
        and department_match
    ):
        full_correct += 1

    total_latency += record["latency_seconds"]


print("TF-IDF Majority Vote + LLM Evaluation 4")
print("---------------------------------------")
print("Total:", total)

print(
    f"Request Type ID Accuracy: {request_type_correct}/{total} "
    f"({request_type_correct / total * 100:.1f}%)"
)

print(
    f"Issue Accuracy: {issue_correct}/{total} "
    f"({issue_correct / total * 100:.1f}%)"
)

print(
    f"Category Accuracy: {category_correct}/{total} "
    f"({category_correct / total * 100:.1f}%)"
)

print(
    f"Department Accuracy: {department_correct}/{total} "
    f"({department_correct / total * 100:.1f}%)"
)

print(
    f"Full Record Accuracy: {full_correct}/{total} "
    f"({full_correct / total * 100:.1f}%)"
)

print(
    f"Average Latency: {total_latency / total:.2f} seconds"
)