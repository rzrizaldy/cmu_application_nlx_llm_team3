import json
import random
from pathlib import Path

input_path = Path("data/pittsburgh311/corpus.jsonl")
output_dir = Path("data/pittsburgh311/dataset_splits")

output_dir.mkdir(parents=True, exist_ok=True)

with input_path.open("r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

random.seed(42)
random.shuffle(records)

split_index = int(len(records) * 0.60)

development = records[:split_index]
evaluation = records[split_index:]

dev_path = output_dir / "development_dataset.jsonl"
eval_path = output_dir / "evaluation_dataset.jsonl"

with dev_path.open("w", encoding="utf-8") as f:
    for record in development:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

with eval_path.open("w", encoding="utf-8") as f:
    for record in evaluation:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("Total:", len(records))
print("Development:", len(development))
print("Evaluation:", len(evaluation))
print("Saved:", dev_path)
print("Saved:", eval_path)