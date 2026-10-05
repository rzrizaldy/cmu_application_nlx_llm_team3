import json
import time
from collections import Counter

from tfidf_router import retrieve_similar_requests


input_file = "data/pittsburgh311/dataset_splits/evaluation_50_text_inputs.jsonl"
output_file = "outputs/tfidf_vote_eval4_responses.jsonl"


records = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip() != "":
            records.append(json.loads(line))


with open(output_file, "w", encoding="utf-8") as out_file:

    for i, record in enumerate(records, start=1):

        start_time = time.perf_counter()

        results = retrieve_similar_requests(
            record["input"],
            top_k=3
        )

        # Count which request type appears most often
        request_type_ids = []

        for result in results:
            request_type_ids.append(result["request_type_id"])

        counts = Counter(request_type_ids)

        majority_request_type = counts.most_common(1)[0][0]

        # Find the highest-ranked result with that request type
        prediction = None

        for result in results:
            if result["request_type_id"] == majority_request_type:
                prediction = {
                    "request_type_id": result["request_type_id"],
                    "issue": result["issue"],
                    "category": result["category"],
                    "department": result["department"]
                }
                break

        latency = time.perf_counter() - start_time

        output = {
            "evaluation": "tfidf_vote_eval4",

            "doc_id": record["doc_id"],
            "input": record["input"],

            "ground_truth": {
                "request_type_id": record["request_type_id"],
                "issue": record["issue"],
                "category": record["category"],
                "department": record["department"]
            },

            "top_3_results": results,

            "prediction": prediction,

            "latency_seconds": latency
        }

        out_file.write(
            json.dumps(output, ensure_ascii=False) + "\n"
        )

        print(
            f"[{i}/50] {record['doc_id']} "
            f"predicted={prediction['issue']} "
            f"({latency:.4f}s)"
        )


print()
print("Finished.")
print("Saved responses to:", output_file)