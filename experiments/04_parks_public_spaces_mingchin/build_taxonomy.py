import json

input_file = "data/pittsburgh311/dataset_splits/development_dataset.jsonl"
output_file = "data/pittsburgh311/taxonomy.json"

taxonomy = {}

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip() != "":
            record = json.loads(line)

            metadata = record["metadata"]

            issue = metadata.get("issue")

            if issue:
                taxonomy[issue] = {
                    "request_type_id": metadata.get("request_type_id"),
                    "issue": issue,
                    "category": metadata.get("category"),
                    "department": metadata.get("department")
                }

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(taxonomy, f, indent=2)

print("Unique issues:", len(taxonomy))
print("Saved:", output_file)