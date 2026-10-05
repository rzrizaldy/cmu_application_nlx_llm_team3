import json
import time

from tfidf_router import retrieve_similar_requests


input_file = "data/pittsburgh311/dataset_splits/evaluation_50_text_inputs.jsonl"
output_file = "outputs/tfidf_eval3_responses.jsonl"


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

        latency = time.perf_counter() - start_time

        top_result = results[0]

        result = {
            "evaluation": "tfidf_eval3",

            "doc_id": record["doc_id"],
            "input": record["input"],

            "ground_truth": {
                "request_type_id": record["request_type_id"],
                "issue": record["issue"],
                "category": record["category"],
                "department": record["department"]
            },

            "top_3_results": results,

            "prediction": {
                "request_type_id": top_result["request_type_id"],
                "issue": top_result["issue"],
                "category": top_result["category"],
                "department": top_result["department"]
            },

            "latency_seconds": latency
        }

        out_file.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )

        print(
            f"[{i}/50] {record['doc_id']} "
            f"predicted={top_result['issue']} "
            f"({latency:.4f}s)"
        )


print()
print("Finished.")
print("Saved responses to:", output_file)