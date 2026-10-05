import json
import time

from tfidf_router import retrieve_similar_requests
from safety_guardrail import guarded_route


adversarial_file = "data/pittsburgh311/adversarial_probes.jsonl"
benign_file = "data/pittsburgh311/benign_safety_inputs.jsonl"


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() != "":
                records.append(json.loads(line))

    return records


records = load_jsonl(adversarial_file) + load_jsonl(benign_file)


baseline_latencies = []
guardrail_latencies = []


for record in records:
    user_input = record["input"]

    # Without guardrail
    start = time.perf_counter()

    retrieve_similar_requests(
        user_input,
        top_k=3
    )

    baseline_latency = time.perf_counter() - start
    baseline_latencies.append(baseline_latency)

    # With guardrail
    result = guarded_route(user_input)
    guardrail_latencies.append(result["latency_seconds"])


avg_baseline = sum(baseline_latencies) / len(baseline_latencies)
avg_guardrail = sum(guardrail_latencies) / len(guardrail_latencies)

added_latency = avg_guardrail - avg_baseline


print("Guardrail Cost")
print("--------------")

print(
    f"Average latency without guardrail: "
    f"{avg_baseline:.6f} seconds"
)

print(
    f"Average latency with guardrail: "
    f"{avg_guardrail:.6f} seconds"
)

print(
    f"Added latency per request: "
    f"{added_latency:.6f} seconds"
)

print("Added tokens per request: 0")