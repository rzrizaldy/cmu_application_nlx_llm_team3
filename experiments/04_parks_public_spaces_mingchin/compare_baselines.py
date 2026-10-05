import json
import re


def normalize_text(value):
    if value is None:
        return ""

    return str(value).strip().lower()


def parse_model_response(response):
    result = {
        "request_type_id": "",
        "issue": "",
        "category": "",
        "department": ""
    }

    patterns = {
        "request_type_id": r"request_type_id\s*:\s*(.+)",
        "issue": r"issue\s*:\s*(.+)",
        "category": r"category\s*:\s*(.+)",
        "department": r"department\s*:\s*(.+)"
    }

    for field in patterns:
        match = re.search(
            patterns[field],
            response,
            flags=re.IGNORECASE
        )

        if match:
            result[field] = match.group(1).strip()

    return result


def evaluate_file(file_path):
    records = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    fields = [
        "request_type_id",
        "issue",
        "category",
        "department"
    ]

    correct_counts = {
        "request_type_id": 0,
        "issue": 0,
        "category": 0,
        "department": 0
    }

    full_correct = 0
    total_latency = 0

    for record in records:

        prediction = parse_model_response(
            record["model_response"]
        )

        truth = record["ground_truth"]

        all_correct = True

        for field in fields:

            predicted_value = normalize_text(
                prediction[field]
            )

            true_value = normalize_text(
                truth[field]
            )

            if predicted_value == true_value:
                correct_counts[field] += 1
            else:
                all_correct = False

        if all_correct:
            full_correct += 1

        total_latency += record["latency_seconds"]

    total = len(records)

    results = {
        "total": total,
        "request_type_id_accuracy":
            correct_counts["request_type_id"] / total,
        "issue_accuracy":
            correct_counts["issue"] / total,
        "category_accuracy":
            correct_counts["category"] / total,
        "department_accuracy":
            correct_counts["department"] / total,
        "full_accuracy":
            full_correct / total,
        "average_latency":
            total_latency / total
    }

    return results


eval1 = evaluate_file(
    "outputs/baseline_eval1_responses.jsonl"
)

eval3 = evaluate_file(
    "outputs/baseline_eval3_responses.jsonl"
)


print("Baseline Evaluation 1")
print("---------------------")
print(
    "Request Type ID Accuracy:",
    f"{eval1['request_type_id_accuracy']:.2%}"
)
print(
    "Issue Accuracy:",
    f"{eval1['issue_accuracy']:.2%}"
)
print(
    "Category Accuracy:",
    f"{eval1['category_accuracy']:.2%}"
)
print(
    "Department Accuracy:",
    f"{eval1['department_accuracy']:.2%}"
)
print(
    "Full Record Accuracy:",
    f"{eval1['full_accuracy']:.2%}"
)
print(
    "Average Latency:",
    f"{eval1['average_latency']:.2f} seconds"
)


print("\nBaseline Evaluation 3")
print("---------------------")
print(
    "Request Type ID Accuracy:",
    f"{eval3['request_type_id_accuracy']:.2%}"
)
print(
    "Issue Accuracy:",
    f"{eval3['issue_accuracy']:.2%}"
)
print(
    "Category Accuracy:",
    f"{eval3['category_accuracy']:.2%}"
)
print(
    "Department Accuracy:",
    f"{eval3['department_accuracy']:.2%}"
)
print(
    "Full Record Accuracy:",
    f"{eval3['full_accuracy']:.2%}"
)
print(
    "Average Latency:",
    f"{eval3['average_latency']:.2f} seconds"
)