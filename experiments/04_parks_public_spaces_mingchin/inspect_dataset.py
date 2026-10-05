import json
from collections import Counter

path = "data/pittsburgh311/dataset_splits/development_dataset.jsonl"

records = []

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

print("Total development records:", len(records))

print("\nModalities:")
print(Counter(r.get("modality") for r in records))

print("\nraw_text missing:")
print(sum(r.get("raw_text") is None for r in records))

print("\ntable_json missing:")
print(sum(r.get("table_json") is None for r in records))

print("\nMetadata fields:")
metadata_keys = set()
for r in records:
    metadata_keys.update(r.get("metadata", {}).keys())

print(sorted(metadata_keys))

print("\nFirst 3 records:")
for i, r in enumerate(records[:3], start=1):
    print(f"\n--- Record {i} ---")
    print("doc_id:", r.get("doc_id"))
    print("modality:", r.get("modality"))
    print("raw_text:", r.get("raw_text"))
    print("table_json:", r.get("table_json"))
    print("metadata:", r.get("metadata"))