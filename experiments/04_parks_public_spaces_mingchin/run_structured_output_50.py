import json
import time

from omegaconf import OmegaConf
from src.generation import GenerationManager
from src.pydantic_models.pittsburgh311 import Pittsburgh311Response


input_file = "data/pittsburgh311/dataset_splits/evaluation_50_text_inputs.jsonl"
output_file = "outputs/structured_output_eval1_responses.jsonl"
schema_file = "data/pydantic_models/pittsburgh311_schema.json"


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

    "structured_output": {
        "enabled": True,
        "schema_path": schema_file,
        "strict": True
    },

    "system_prompt": (
        "You are assisting with Pittsburgh 311 service request routing."
    )
})


generator = GenerationManager()

print("Loading Phi-4-mini-instruct...")

model, tokenizer, device = generator._load_model_and_tokenizer(cfg)

print("Model loaded on:", device)


records = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip() != "":
            records.append(json.loads(line))


with open(schema_file, "r", encoding="utf-8") as f:
    schema_text = f.read()


schema_instruction = (
    "\n\nRespond with ONLY a single JSON object that strictly matches "
    "this JSON Schema, with no other text:\n"
    + schema_text
)

system_message = cfg.system_prompt + schema_instruction


with open(output_file, "w", encoding="utf-8") as out_file:

    for i, record in enumerate(records, start=1):

        prompt = (
            "Classify this Pittsburgh 311 resident request.\n\n"
            f"Resident request: {record['input']}"
        )

        messages = [
            {
                "role": "system",
                "content": system_message
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

        valid_json = True
        schema_valid = True
        parsed_response = None

        try:
            parsed_response = json.loads(response)
        except json.JSONDecodeError:
            valid_json = False
            schema_valid = False

        if valid_json:
            try:
                Pittsburgh311Response.model_validate(parsed_response)
            except Exception:
                schema_valid = False

        result = {
            "evaluation": "structured_output_eval1",
            "doc_id": record["doc_id"],
            "input": record["input"],

            "ground_truth": {
                "request_type_id": record["request_type_id"],
                "issue": record["issue"],
                "category": record["category"],
                "department": record["department"]
            },

            "model_response": response,
            "parsed_response": parsed_response,
            "valid_json": valid_json,
            "schema_valid": schema_valid,

            "generation_settings": {
                "temperature": 0.2,
                "top_p": 0.8,
                "max_new_tokens": 128
            },

            "latency_seconds": latency
        }

        out_file.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )

        out_file.flush()

        print(
            f"[{i}/50] {record['doc_id']} "
            f"valid_json={valid_json} "
            f"schema_valid={schema_valid} "
            f"({latency:.2f}s)"
        )


print()
print("Finished.")
print("Saved responses to:", output_file)