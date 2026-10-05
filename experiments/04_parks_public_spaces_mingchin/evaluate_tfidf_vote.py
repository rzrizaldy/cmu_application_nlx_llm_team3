import json


input_file = "outputs/tfidf_vote_eval4_responses.jsonl"

records = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip() != "":
            records.append(json.loads(line))


def clean(value):
    if value is None:
        return ""
    return str(value).strip().lower()


total = len(records)

request_type_correct = 0
issue_correct = 0
category_correct = 0
department_correct = 0
full_correct = 0

total_latency = 0


for record in records:
    ground_truth = record["ground_truth"]
    prediction = record["prediction"]

    total_latency += record["latency_seconds"]

    request_match = (
        clean(prediction.get("request_type_id"))
        == clean(ground_truth.get("request_type_id"))
    )

    issue_match = (
        clean(prediction.get("issue"))
        == clean(ground_truth.get("issue"))
    )

    category_match = (
        clean(prediction.get("category"))
        == clean(ground_truth.get("category"))
    )

    department_match = (
        clean(prediction.get("department"))
        == clean(ground_truth.get("department"))
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


print("TF-IDF Majority Vote Evaluation")
print("-------------------------------")
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
    f"Average Latency: {total_latency / total:.4f} seconds"
)