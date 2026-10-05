import json


input_file = "outputs/structured_output_eval1_responses.jsonl"

records = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip() != "":
            records.append(json.loads(line))


total = len(records)

valid_json_count = 0
schema_valid_count = 0

request_type_correct = 0
issue_correct = 0
category_correct = 0
department_correct = 0
full_record_correct = 0

total_latency = 0


for record in records:

    if record["valid_json"]:
        valid_json_count += 1

    if record["schema_valid"]:
        schema_valid_count += 1

    prediction = record["parsed_response"]
    truth = record["ground_truth"]

    all_correct = True

    if prediction is None:
        all_correct = False

    else:

        pred_request_type = str(
            prediction.get("request_type_id", "")
        ).strip().lower()

        pred_issue = str(
            prediction.get("issue", "")
        ).strip().lower()

        pred_category = str(
            prediction.get("category", "")
        ).strip().lower()

        pred_department = str(
            prediction.get("department", "")
        ).strip().lower()


        true_request_type = str(
            truth["request_type_id"]
        ).strip().lower()

        true_issue = str(
            truth["issue"]
        ).strip().lower()

        true_category = str(
            truth["category"]
        ).strip().lower()

        true_department = str(
            truth["department"]
        ).strip().lower()


        if pred_request_type == true_request_type:
            request_type_correct += 1
        else:
            all_correct = False

        if pred_issue == true_issue:
            issue_correct += 1
        else:
            all_correct = False

        if pred_category == true_category:
            category_correct += 1
        else:
            all_correct = False

        if pred_department == true_department:
            department_correct += 1
        else:
            all_correct = False


    if all_correct:
        full_record_correct += 1

    total_latency += record["latency_seconds"]


print("Total records:", total)

print(
    "Valid JSON:",
    f"{valid_json_count}/{total}",
    f"= {valid_json_count / total:.2%}"
)

print(
    "Schema valid:",
    f"{schema_valid_count}/{total}",
    f"= {schema_valid_count / total:.2%}"
)

print()

print(
    "Request Type ID Accuracy:",
    f"{request_type_correct / total:.2%}"
)

print(
    "Issue Accuracy:",
    f"{issue_correct / total:.2%}"
)

print(
    "Category Accuracy:",
    f"{category_correct / total:.2%}"
)

print(
    "Department Accuracy:",
    f"{department_correct / total:.2%}"
)

print(
    "Full Record Accuracy:",
    f"{full_record_correct / total:.2%}"
)

print(
    "Average Latency:",
    f"{total_latency / total:.2f} seconds"
)