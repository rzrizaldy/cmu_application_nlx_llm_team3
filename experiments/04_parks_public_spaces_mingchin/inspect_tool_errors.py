import json


input_file = "outputs/tool_calling_eval2_responses.jsonl"

records = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip() != "":
            records.append(json.loads(line))


def clean(value):
    if value is None:
        return ""
    return str(value).strip().lower()


error_count = 0

for record in records:
    ground_truth = record["ground_truth"]
    tool_result = record.get("tool_result")

    if not isinstance(tool_result, dict):
        continue

    if "error" in tool_result:
        error_count += 1

        print("=" * 80)
        print("Doc ID:", record["doc_id"])
        print("Input:", record["input"])
        print("Tool result:", tool_result)
        print("Ground truth:", ground_truth)
        print()

        continue

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

    full_match = (
        request_match
        and issue_match
        and category_match
        and department_match
    )

    if not full_match:
        error_count += 1

        print("=" * 80)
        print("Doc ID:", record["doc_id"])
        print("Input:", record["input"])

        print("\nGround truth:")
        print(ground_truth)

        print("\nTool result:")
        print(tool_result)

        print("\nMatches:")
        print("request_type_id:", request_match)
        print("issue:", issue_match)
        print("category:", category_match)
        print("department:", department_match)

        print()


print("Total error cases:", error_count)