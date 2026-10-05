import json
import random
from pathlib import Path

input_path = Path(
    "data/pittsburgh311/dataset_splits/development_dataset.jsonl"
)

output_path = Path(
    "data/pittsburgh311/dataset_splits/baseline_50_text_inputs.jsonl"
)

records = []

with input_path.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        record = json.loads(line)

        if record.get("modality") == "text" and record.get("raw_text"):
            records.append(record)

print("Available text records:", len(records))

random.seed(42)
selected = random.sample(records, 50)

with output_path.open("w", encoding="utf-8") as f:
    for record in selected:
        output_record = {
            "doc_id": record["doc_id"],
            "input": record["raw_text"],
            "request_type_id": record["metadata"].get("request_type_id"),
            "issue": record["metadata"].get("issue"),
            "category": record["metadata"].get("category"),
            "department": record["metadata"].get("department"),
        }

        f.write(json.dumps(output_record, ensure_ascii=False) + "\n")

print("Selected:", len(selected))
print("Saved:", output_path)