import json
import time
from collections import Counter

from omegaconf import OmegaConf

from src.generation import GenerationManager
from tfidf_router import retrieve_similar_requests


input_file = "data/pittsburgh311/dataset_splits/evaluation_50_text_inputs.jsonl"
output_file = "outputs/tfidf_vote_llm_eval4_responses.jsonl"


cfg = OmegaConf.create({
    "model": {
        "name": "phi4_instruct",
        "source": "local",
        "model_id": "microsoft/Phi-4-mini-instruct",
        "local_path": "./models/llms/microsoft/phi-4-mini-instruct",
        "architecture": "phi3",
        "dtype": "bfloat16",
        "device": "auto",
        "trust_remote_code": False,
        "supports_tool_calling": True,
        "supports_structured_output": True,
        "tool_calling_format": "functools_prompt",
        "chat_template_kwargs": {}
    },

    "generation": {
        "max_new_tokens": 128,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 64,
        "do_sample": True,
        "repetition_penalty": 1.0
    },

    "system_prompt": (
        "You are assisting with Pittsburgh 311 service request routing. "
        "Use the provided similar examples and majority result as references."
    )
})


records = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))


generator = GenerationManager()

print("Loading model...")

model, tokenizer, device = generator._load_model_and_tokenizer(cfg)

print("Model loaded on:", device)


with open(output_file, "w", encoding="utf-8") as out_file:

    for i, record in enumerate(records, start=1):

        retrieved = retrieve_similar_requests(
            record["input"],
            top_k=3
        )

        request_type_ids = []

        for result in retrieved:
            request_type_ids.append(result["request_type_id"])

        counts = Counter(request_type_ids)
        majority_id = counts.most_common(1)[0][0]

        majority_result = None

        for result in retrieved:
            if result["request_type_id"] == majority_id:
                majority_result = result
                break


        examples_text = ""

        for j, result in enumerate(retrieved, start=1):
            examples_text += f"""
Example {j}:
Request: {result["example_request"]}
request_type_id: {result["request_type_id"]}
issue: {result["issue"]}
category: {result["category"]}
department: {result["department"]}
"""


        prompt = f"""
Here are three similar Pittsburgh 311 examples:

{examples_text}

The majority routing result is:
request_type_id: {majority_result["request_type_id"]}
issue: {majority_result["issue"]}
category: {majority_result["category"]}
department: {majority_result["department"]}

Now classify this new resident request:

{record["input"]}

Return exactly these four fields:
request_type_id:
issue:
category:
department:
"""

        messages = [
            {
                "role": "system",
                "content": cfg.system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        start_time = time.perf_counter()

        response = generator._generate_once(
            model,
            tokenizer,
            device,
            messages,
            cfg
        )

        latency = time.perf_counter() - start_time


        output = {
            "evaluation": "tfidf_vote_llm_eval4",

            "doc_id": record["doc_id"],
            "input": record["input"],

            "ground_truth": {
                "request_type_id": record["request_type_id"],
                "issue": record["issue"],
                "category": record["category"],
                "department": record["department"]
            },

            "top_3_results": retrieved,
            "majority_result": majority_result,
            "model_response": response,
            "latency_seconds": latency
        }

        out_file.write(
            json.dumps(output, ensure_ascii=False) + "\n"
        )

        out_file.flush()

        print(
            f"[{i}/50] {record['doc_id']} "
            f"majority={majority_result['issue']} "
            f"({latency:.2f}s)"
        )


print()
print("Finished.")
print("Saved responses to:", output_file)