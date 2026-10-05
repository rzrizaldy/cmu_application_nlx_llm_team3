import json


input_file = "outputs/tool_calling_eval2_responses.jsonl"

records = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip() != "":
            records.append(json.loads(line))


total = len(records)

tool_used_count = 0
tool_success_count = 0

request_type_correct = 0
issue_correct = 0
category_correct = 0
department_correct = 0
full_correct = 0

total_latency = 0


def clean(value):
    if value is None:
        return ""
    return str(value).strip().lower()


for record in records:
    total_latency += record["latency_seconds"]

    if record["tool_used"]:
        tool_used_count += 1

    tool_result = record.get("tool_result")

    if isinstance(tool_result, dict) and "error" not in tool_result:
        tool_success_count += 1

        ground_truth = record["ground_truth"]

        request_match = (
            clean(tool_result.get("request_type_id"))
            == clean(ground_truth.get("request_type_id"))
        )

        issue_match = (
            clean(tool_result.get("issue"))
            == clean(ground_truth.get("issue"))
        )

        category_match = (
            clean(tool_result.get("category"))
            == clean(ground_truth.get("category"))
        )

        department_match = (
            clean(tool_result.get("department"))
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


print("Tool Calling Evaluation")
print("-----------------------")
print("Total:", total)

print(
    f"Tool Usage Rate: {tool_used_count}/{total} "
    f"({tool_used_count / total * 100:.1f}%)"
)

print(
    f"Tool Lookup Success Rate: {tool_success_count}/{total} "
    f"({tool_success_count / total * 100:.1f}%)"
)

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