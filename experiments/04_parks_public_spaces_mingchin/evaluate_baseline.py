import json
import re
from pathlib import Path

INPUT_PATH = Path("outputs/baseline_eval1_responses.jsonl")


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def parse_model_response(response):
    parsed = {
        "request_type_id": "",
        "issue": "",
        "category": "",
        "department": "",
    }

    patterns = {
        "request_type_id": r"request_type_id\s*:\s*(.+)",
        "issue": r"issue\s*:\s*(.+)",
        "category": r"category\s*:\s*(.+)",
        "department": r"department\s*:\s*(.+)",
    }

    for field, pattern in patterns.items():
        match = re.search(
            pattern,
            response,
            flags=re.IGNORECASE
        )

        if match:
            parsed[field] = match.group(1).strip()

    return parsed


def main():
    records = []

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    fields = [
        "request_type_id",
        "issue",
        "category",
        "department",
    ]

    correct_counts = {
        field: 0
        for field in fields
    }

    full_record_correct = 0
    total_latency = 0

    for record in records:
        prediction = parse_model_response(
            record["model_response"]
        )

        ground_truth = record["ground_truth"]

        all_correct = True

        for field in fields:
            predicted_value = normalize_text(
                prediction[field]
            )

            true_value = normalize_text(
                ground_truth[field]
            )

            is_correct = predicted_value == true_value

            if is_correct:
                correct_counts[field] += 1
            else:
                all_correct = False

        if all_correct:
            full_record_correct += 1

        total_latency += record["latency_seconds"]

    total = len(records)

    print("Total records:", total)
    print()

    for field in fields:
        accuracy = correct_counts[field] / total
        print(
            f"{field} accuracy: "
            f"{correct_counts[field]}/{total} "
            f"= {accuracy:.2%}"
        )

    full_accuracy = full_record_correct / total

    print()
    print(
        f"Full-record accuracy: "
        f"{full_record_correct}/{total} "
        f"= {full_accuracy:.2%}"
    )

    average_latency = total_latency / total

    print(
        f"Average latency: "
        f"{average_latency:.2f} seconds"
    )


if __name__ == "__main__":
    main()